mod f2;
mod fast_hash;
mod memory_probe;
mod milnor;
mod resolution;
mod sharded_io;
mod subalgebra;

use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use memory_probe::{MemoryMonitorGuard, ProcessMemory, bytes_to_gib, bytes_to_gib_signed};
use milnor::Milnor;
use resolution::{
    CachePolicy, ComputeCursor, ComputeMode, ComputeProgress, E0CacheScope, E0EmptyCachePolicy,
    FixedTBatchScheduler, Resolution, SignatureMatrixCache, set_allocator_relief_enabled,
};
use subalgebra::{
    Subalgebra, SubalgebraSelectionMode, detailed_ext_table_coverages,
    set_subalgebra_selection_mode,
};

const DEFAULT_EXPORT_PATH: &str = "exports/current_basis.csv";
const DEFAULT_SHARDED_CHECKPOINT_PATH: &str = "nassau_min_res.sharded";
const DEFAULT_CACHE_WINDOW_T: usize = 8;
const MAX_DEFAULT_FIXED_T_BATCH_WORKERS: usize = 4;
const FULL_NAIVE_BATCH_DIAGNOSTIC_MAX_T: usize = 64;
const MEMORY_PRESSURE_COMPRESSED_WARN_BYTES: u64 = 512 * 1024 * 1024;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print_help();
        return Ok(());
    }

    let cmd = args.remove(0);
    match cmd.as_str() {
        "compute" => compute_cmd(&args),
        "checkpoint" => checkpoint_cmd(&args),
        "multiply" => multiply_cmd(&args),
        "basis" => basis_cmd(&args),
        "range" => range_cmd(&args),
        "export" | "export-csv" => export_cmd(&args),
        "detailed-subalgebras"
        | "detailed_subalgebras"
        | "detailed-ext-tables"
        | "detailed_ext_tables" => detailed_subalgebras_cmd(&args),
        _ => Err(format!("unknown command `{cmd}`; run with --help")),
    }
}

fn compute_cmd(args: &[String]) -> Result<(), String> {
    validate_flags(args, COMPUTE_FLAGS)?;
    if has_flag(args, "-h") || has_flag(args, "--help") {
        print_compute_help();
        return Ok(());
    }

    let total_timer = Instant::now();
    let t_flag = parse_usize_flag(args, "--t")?;
    let max_t = t_flag.unwrap_or(12);
    let output = parse_string_flag_any(args, &["--output", "--out"])?.map(PathBuf::from);
    let show_differentials = has_flag(args, "--show-differentials");
    let show_steps = has_flag(args, "--show-steps");
    let checkpoint_paths = checkpoint_paths_for(args, output.as_ref(), max_t)?;
    let fresh = has_flag(args, "--fresh");
    let verbose = has_flag(args, "--verbose");
    let checkpoint_schedule = checkpoint_schedule_for(args)?;
    let cache_prune_min_t = parse_usize_flag(args, "--cache-prune-min-t")?;
    let cache_window_t_flag = parse_usize_flag(args, "--cache-window-t")?;
    if cache_window_t_flag.is_some() && cache_prune_min_t.is_none() {
        return Err("--cache-window-t only has an effect with --cache-prune-min-t".to_string());
    }
    let cache_window_t = cache_window_t_flag.unwrap_or(DEFAULT_CACHE_WINDOW_T);
    let e0_cache_scope = parse_e0_cache_scope(args)?;
    let e0_empty_cache = parse_e0_empty_cache_policy(args)?;
    let signature_matrix_cache = parse_signature_matrix_cache(args)?;
    let allocator_relief = parse_allocator_relief(args)?;
    configure_subalgebra_selection(args)?;
    configure_rayon_threads(args)?;
    let mode = build_compute_mode(args, max_t)?;
    let rayon_threads = rayon::current_num_threads();
    set_allocator_relief_enabled(allocator_relief);
    if verbose {
        print_startup_info(
            max_t,
            output.as_ref(),
            checkpoint_paths.load.as_ref(),
            checkpoint_paths.save.as_ref(),
            rayon_threads,
        );
        print_checkpoint_paths(
            checkpoint_paths.load.as_ref(),
            checkpoint_paths.save.as_ref(),
        );
        print_cache_settings(
            cache_prune_min_t,
            cache_window_t,
            e0_cache_scope,
            e0_empty_cache,
            signature_matrix_cache,
            allocator_relief,
        );
    }

    let (mut res, start_cursor) = load_or_start(
        max_t,
        checkpoint_paths.load.as_ref(),
        checkpoint_paths.load_required,
        fresh,
        verbose,
    )?;
    res.set_cache_policy(CachePolicy {
        e0_cache_scope,
        signature_matrix_cache,
        e0_empty_cache,
    });
    if start_cursor.next_t > max_t + 1 {
        let completed_t = start_cursor.next_t.saturating_sub(1);
        let source = checkpoint_display(checkpoint_paths.load.as_ref());
        return Err(format!(
            "input checkpoint {source} is already complete through t={completed_t}, but this run requested --t {max_t}; use --t >= {completed_t}, choose a smaller checkpoint, or use --fresh"
        ));
    }
    let _memory_monitor = MemoryMonitorGuard::start_from_env();
    let compute_timer = Instant::now();
    let mut final_cursor = start_cursor;
    let mut last_checkpoint_t = start_cursor.next_t.saturating_sub(1);
    let mut layer_t = start_cursor.next_t;
    let mut layer_timer = Instant::now();
    let mut layer_usage = ProcessUsage::now();
    final_cursor =
        res.compute_from_cursor_with_progress(max_t, mode, start_cursor, |res, progress| {
            if progress.t != layer_t {
                layer_t = progress.t;
                layer_timer = Instant::now();
                layer_usage = ProcessUsage::now();
            }
            let cursor = ComputeCursor {
                next_t: progress.next_t,
                next_s: progress.next_s,
            };
            final_cursor = cursor;

            let completed_layer = cursor.next_s == 0;
            if completed_layer {
                let layer_cpu = ProcessUsage::now()
                    .and_then(|usage| layer_usage.map(|start| usage.saturating_since(start)));
                print_layer_status(
                    total_timer,
                    progress,
                    layer_timer.elapsed(),
                    layer_cpu,
                    verbose,
                );
                let completed_t = cursor.next_t.saturating_sub(1);
                if cache_prune_min_t
                    .map(|min_t| completed_t >= min_t)
                    .unwrap_or(false)
                {
                    res.prune_t_caches(cursor.next_t, cache_window_t);
                }
            }

            if !cursor.is_complete_for(max_t)
                && checkpoint_is_due(checkpoint_schedule, cursor, last_checkpoint_t)
            {
                save_checkpoint_if_enabled(checkpoint_paths.save.as_ref(), res, cursor, verbose)?;
                last_checkpoint_t = cursor.next_t.saturating_sub(1);
            }
            if completed_layer {
                layer_t = cursor.next_t;
                layer_timer = Instant::now();
                layer_usage = ProcessUsage::now();
            }
            Ok(())
        })?;
    save_checkpoint_if_enabled(checkpoint_paths.save.as_ref(), &res, final_cursor, verbose)?;
    let compute_s = compute_timer.elapsed().as_secs_f64();

    if let Some(path) = output {
        let report = res.report(max_t, show_differentials, show_steps);
        std::fs::write(&path, &report)
            .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
        println!("wrote {}", path.display());
        println!("computed t <= {max_t}, s <= {max_t}");
        if show_steps {
            println!("included per-bidegree Algorithm 2/Algorithm 1 log");
        } else {
            println!("use --show-steps to include the per-bidegree Algorithm 2/Algorithm 1 log");
        }
    } else if has_flag(args, "--print-report") {
        let report = res.report(max_t, show_differentials, show_steps);
        print!("{report}");
    } else {
        println!(
            "computed t <= {max_t}, s <= {}, generators={}",
            max_t,
            res.generator_count()
        );
    }
    if verbose {
        eprintln!(
            "timing nassau_min_res compute_s={compute_s:.6} total_s={:.6}",
            total_timer.elapsed().as_secs_f64()
        );
    } else {
        eprintln!("finished in {:.3}s", total_timer.elapsed().as_secs_f64());
    }
    Ok(())
}

fn build_compute_mode(args: &[String], max_t: usize) -> Result<ComputeMode, String> {
    let algorithm =
        parse_string_flag(args, "--algorithm")?.unwrap_or_else(|| "fixed-t-batch".to_string());
    match algorithm.as_str() {
        "auto" | "sequential" | "nassau-auto" | "algorithm2-auto" | "alg2-auto" => {
            let subalgebras = parse_subalgebra_list(args, max_t)?;
            let strict = has_flag(args, "--strict");
            let force = has_flag(args, "--force");
            Ok(ComputeMode::Auto {
                subalgebras,
                strict,
                force,
            })
        }
        "fixed-t-batch" | "fixed_t_batch" => {
            let workers = Some(
                parse_usize_flag(args, "--batch-workers")?
                    .unwrap_or_else(default_fixed_t_batch_workers),
            );
            let scheduler = parse_batch_scheduler(args)?;
            let validate_commit = parse_batch_commit_check(args)?;
            let inner = build_fixed_t_batch_inner_mode(args, max_t)?;
            if matches!(workers, Some(0)) {
                return Err("--batch-workers expects a positive integer".to_string());
            }
            Ok(ComputeMode::FixedTBatch {
                workers,
                shadow: false,
                validate_commit,
                scheduler,
                inner: Box::new(inner),
            })
        }
        "fixed-t-batch-shadow" | "fixed_t_batch_shadow" => {
            let workers = Some(
                parse_usize_flag(args, "--batch-workers")?
                    .unwrap_or_else(default_fixed_t_batch_workers),
            );
            let scheduler = parse_batch_scheduler(args)?;
            let validate_commit = parse_batch_commit_check(args)?;
            let inner = build_fixed_t_batch_inner_mode(args, max_t)?;
            if matches!(workers, Some(0)) {
                return Err("--batch-workers expects a positive integer".to_string());
            }
            Ok(ComputeMode::FixedTBatch {
                workers,
                shadow: true,
                validate_commit,
                scheduler,
                inner: Box::new(inner),
            })
        }
        "nassau" | "algorithm2" | "alg2" => {
            let subalgebra = parse_string_flag(args, "--subalgebra")?
                .ok_or_else(|| {
                    "Algorithm 2 needs an explicit --subalgebra A0/A1/A2/...".to_string()
                })
                .and_then(|name| Subalgebra::parse(&name, max_t))?;
            let strict = has_flag(args, "--strict");
            let force = has_flag(args, "--force");
            Ok(ComputeMode::Nassau {
                subalgebra,
                strict,
                force,
            })
        }
        "naive" | "algorithm1" | "alg1" => Ok(ComputeMode::Naive),
        _ => Err(
            "--algorithm must be sequential, auto, nassau, naive, fixed-t-batch, or fixed-t-batch-shadow"
                .into(),
        ),
    }
}

fn default_fixed_t_batch_workers() -> usize {
    default_fixed_t_batch_workers_for_threads(rayon::current_num_threads())
}

fn default_fixed_t_batch_workers_for_threads(rayon_threads: usize) -> usize {
    (rayon_threads / 2).clamp(1, MAX_DEFAULT_FIXED_T_BATCH_WORKERS)
}

fn build_fixed_t_batch_inner_mode(args: &[String], max_t: usize) -> Result<ComputeMode, String> {
    let inner = parse_string_flag(args, "--batch-inner")?.unwrap_or_else(|| "auto".to_string());
    match inner.as_str() {
        "auto" | "sequential" | "nassau-auto" | "algorithm2-auto" | "alg2-auto" => {
            let subalgebras = parse_subalgebra_list(args, max_t)?;
            let force = has_flag(args, "--force");
            let bounded_naive_fallback =
                has_flag(args, "--strict") || max_t > FULL_NAIVE_BATCH_DIAGNOSTIC_MAX_T;
            let mode = if bounded_naive_fallback {
                ComputeMode::AutoBoundedNaiveFallback { subalgebras, force }
            } else {
                ComputeMode::Auto {
                    subalgebras,
                    strict: false,
                    force,
                }
            };
            Ok(mode)
        }
        "nassau" | "algorithm2" | "alg2" => {
            let subalgebra = parse_string_flag(args, "--subalgebra")?
                .ok_or_else(|| {
                    "fixed-t-batch --batch-inner nassau needs --subalgebra A0/A1/A2/...".to_string()
                })
                .and_then(|name| Subalgebra::parse(&name, max_t))?;
            let strict = has_flag(args, "--strict") || max_t > FULL_NAIVE_BATCH_DIAGNOSTIC_MAX_T;
            let force = has_flag(args, "--force");
            Ok(ComputeMode::Nassau {
                subalgebra,
                strict,
                force,
            })
        }
        "naive" | "algorithm1" | "alg1" => {
            if !has_flag(args, "--allow-full-naive-batch") {
                return Err(
                    "--batch-inner naive uses full frozen homology and is memory-prohibitive in high degrees; add --allow-full-naive-batch only for low-dimensional diagnostics"
                        .to_string(),
                );
            }
            if max_t > FULL_NAIVE_BATCH_DIAGNOSTIC_MAX_T {
                return Err(format!(
                    "--batch-inner naive is full frozen homology and is blocked for t > {FULL_NAIVE_BATCH_DIAGNOSTIC_MAX_T}; fixed-t batch experiments must use Nassau/signature reduction via --batch-inner auto or nassau."
                ));
            }
            Ok(ComputeMode::Naive)
        }
        _ => Err(format!(
            "--batch-inner must be auto, nassau, or naive, got `{inner}`"
        )),
    }
}

struct CheckpointPaths {
    load: Option<PathBuf>,
    save: Option<PathBuf>,
    load_required: bool,
}

fn checkpoint_paths_for(
    args: &[String],
    output: Option<&PathBuf>,
    max_t: usize,
) -> Result<CheckpointPaths, String> {
    if has_flag(args, "--no-checkpoint") {
        if parse_string_flag(args, "--checkpoint")?.is_some()
            || parse_string_flag(args, "--load-checkpoint")?.is_some()
            || parse_string_flag(args, "--save-checkpoint")?.is_some()
        {
            return Err(
                "--no-checkpoint cannot be combined with --checkpoint, --load-checkpoint, or --save-checkpoint"
                    .to_string(),
            );
        }
        return Ok(CheckpointPaths {
            load: None,
            save: None,
            load_required: false,
        });
    }

    let legacy_checkpoint = parse_string_flag(args, "--checkpoint")?.map(PathBuf::from);
    let load_checkpoint = parse_string_flag(args, "--load-checkpoint")?.map(PathBuf::from);
    let save_checkpoint = parse_string_flag(args, "--save-checkpoint")?.map(PathBuf::from);
    if legacy_checkpoint.is_some() && (load_checkpoint.is_some() || save_checkpoint.is_some()) {
        return Err(
            "--checkpoint cannot be combined with --load-checkpoint or --save-checkpoint"
                .to_string(),
        );
    }
    let default_checkpoint = output
        .map(|path| default_sharded_checkpoint_path(path.as_path()))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SHARDED_CHECKPOINT_PATH));

    let explicit_load = load_checkpoint.is_some();
    let load_required = explicit_load;
    let load = load_checkpoint
        .or_else(|| legacy_checkpoint.clone())
        .or_else(|| Some(default_checkpoint.clone()));
    let save = save_checkpoint
        .or_else(|| legacy_checkpoint.clone())
        .or_else(|| {
            explicit_load.then(|| PathBuf::from(format!("nassau_min_res_t{max_t}.sharded")))
        })
        .or_else(|| Some(default_checkpoint.clone()));

    Ok(CheckpointPaths {
        load,
        save,
        load_required,
    })
}

fn default_sharded_checkpoint_path(output: &Path) -> PathBuf {
    let mut path = output.as_os_str().to_os_string();
    path.push(".sharded");
    PathBuf::from(path)
}

fn checkpoint_display(path: Option<&PathBuf>) -> String {
    path.map(|path| path.display().to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn print_checkpoint_paths(load_path: Option<&PathBuf>, save_path: Option<&PathBuf>) {
    if load_path != save_path {
        eprintln!(
            "checkpoint nassau_min_res paths load_checkpoint={} save_checkpoint={}",
            checkpoint_display(load_path),
            checkpoint_display(save_path),
        );
    }
}

fn print_startup_info(
    max_t: usize,
    output: Option<&PathBuf>,
    load_checkpoint: Option<&PathBuf>,
    save_checkpoint: Option<&PathBuf>,
    rayon_threads: usize,
) {
    let argv = env::args().collect::<Vec<_>>();
    let cwd = env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let output_path = output
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "none".to_string());
    let output_dir = output
        .and_then(|path| path.parent())
        .filter(|path| !path.as_os_str().is_empty())
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| ".".to_string());
    eprintln!(
        "startup nassau_min_res package={} version={} git_commit={} cwd={} argv={:?}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        option_env!("GIT_COMMIT").unwrap_or("unknown"),
        cwd,
        argv,
    );
    eprintln!(
        "startup nassau_min_res max_t={} max_s={} output={} output_dir={} load_checkpoint={} save_checkpoint={} rayon_threads={} env_RAYON_NUM_THREADS={} env_NASSAU_FULL_DIFFERENTIAL_CHUNK_MB={}",
        max_t,
        max_t,
        output_path,
        output_dir,
        checkpoint_display(load_checkpoint),
        checkpoint_display(save_checkpoint),
        rayon_threads,
        env_value("RAYON_NUM_THREADS"),
        env_value("NASSAU_FULL_DIFFERENTIAL_CHUNK_MB"),
    );
}

fn env_value(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| "unset".to_string())
}

fn source_git_commit() -> Option<String> {
    option_env!("GIT_COMMIT")
        .filter(|value| *value != "unknown")
        .map(str::to_string)
}

fn load_or_start(
    max_t: usize,
    checkpoint_path: Option<&PathBuf>,
    checkpoint_required: bool,
    fresh: bool,
    verbose: bool,
) -> Result<(Resolution, ComputeCursor), String> {
    let Some(path) = checkpoint_path else {
        if verbose {
            eprintln!(
                "checkpoint nassau_min_res start=fresh checkpoint=none reason=no_checkpoint_path"
            );
        }
        return Ok((Resolution::new(max_t), ComputeCursor::start()));
    };

    if fresh {
        if verbose {
            eprintln!(
                "checkpoint nassau_min_res start=fresh checkpoint={} reason=fresh_requested",
                path.display()
            );
        }
        return Ok((Resolution::new(max_t), ComputeCursor::start()));
    }

    if path.exists() {
        let (resolution, cursor) = load_sharded_checkpoint_for_compute(path, max_t)?;
        if cursor.is_complete_for(max_t) {
            if verbose {
                eprintln!(
                    "checkpoint nassau_min_res start=complete checkpoint={} next_t={} next_s={} generators={}",
                    path.display(),
                    cursor.next_t,
                    cursor.next_s,
                    resolution.generator_count()
                );
            } else {
                eprintln!(
                    "loaded checkpoint {} through t={}",
                    path.display(),
                    cursor.next_t.saturating_sub(1)
                );
            }
        } else {
            if verbose {
                eprintln!(
                    "checkpoint nassau_min_res start=resume checkpoint={} next_t={} next_s={} generators={}",
                    path.display(),
                    cursor.next_t,
                    cursor.next_s,
                    resolution.generator_count()
                );
            } else {
                eprintln!("resuming {} at t={}", path.display(), cursor.next_t);
            }
        }
        Ok((resolution, cursor))
    } else if checkpoint_required {
        Err(format!(
            "requested input checkpoint {} does not exist",
            path.display()
        ))
    } else {
        if verbose {
            eprintln!(
                "checkpoint nassau_min_res start=fresh checkpoint={} reason=no_checkpoint",
                path.display()
            );
        }
        Ok((Resolution::new(max_t), ComputeCursor::start()))
    }
}

fn load_sharded_checkpoint_for_compute(
    path: &Path,
    max_t: usize,
) -> Result<(Resolution, ComputeCursor), String> {
    if !path.is_dir() {
        return Err(format!(
            "checkpoint {} is not a directory; local checkpoints are sharded NMR_SHARDED_V1 directories",
            path.display()
        ));
    }
    let manifest = sharded_io::read_manifest(path)?;
    if manifest.format_name != sharded_io::SHARDED_CHECKPOINT_FORMAT {
        return Err(format!(
            "checkpoint directory {} has format `{}`, expected `{}`",
            path.display(),
            manifest.format_name,
            sharded_io::SHARDED_CHECKPOINT_FORMAT
        ));
    }
    let all_qs = (0..=manifest.max_homological_degree).collect::<BTreeSet<_>>();
    let loaded = sharded_io::load_sparse_snapshot(path, &[], &all_qs, &all_qs)?;
    let resolution = Resolution::from_sparse_snapshot(
        max_t.max(manifest.completed_internal_degree),
        loaded.total_generator_count,
        loaded.snapshot,
    )?;
    let cursor = ComputeCursor {
        next_t: manifest.next_internal_degree,
        next_s: 0,
    };
    Ok((resolution, cursor))
}

fn save_checkpoint_if_enabled(
    checkpoint_path: Option<&PathBuf>,
    resolution: &Resolution,
    cursor: ComputeCursor,
    verbose: bool,
) -> Result<(), String> {
    if let Some(path) = checkpoint_path {
        let timer = Instant::now();
        let manifest = sharded_io::write_sharded_checkpoint(
            path,
            resolution,
            cursor,
            None,
            source_git_commit(),
            true,
        )?;
        let size_mb = sharded_io::sharded_tree_size(path)
            .map(|bytes| bytes as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        if verbose {
            eprintln!(
                "checkpoint nassau_min_res saved checkpoint={} format={} elapsed_s={:.3} size_mb={:.1} next_t={} next_s={}",
                path.display(),
                manifest.format_name,
                timer.elapsed().as_secs_f64(),
                size_mb,
                cursor.next_t,
                cursor.next_s,
            );
        } else {
            eprintln!("saved checkpoint {}", path.display());
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum CheckpointSchedule {
    EveryLayers(usize),
    FinalOnly,
}

fn print_cache_settings(
    cache_prune_min_t: Option<usize>,
    cache_window_t: usize,
    e0_cache_scope: E0CacheScope,
    e0_empty_cache: E0EmptyCachePolicy,
    signature_matrix_cache: SignatureMatrixCache,
    allocator_relief: bool,
) {
    let prune = cache_prune_min_t
        .map(|min_t| format!("enabled:min_t={min_t}:window_t={cache_window_t}"))
        .unwrap_or_else(|| "disabled".to_string());
    let allocator_relief = if allocator_relief { "on" } else { "off" };
    eprintln!(
        "cache nassau_min_res invalidation=precise solver_cache=true step_solver_cache=true full_basis_lookup_cache=true algebra_basis_index_cache=true full_differential=chunked cache_pruning={} e0_cache_scope={} e0_empty_cache={} signature_matrix_cache={} allocator_relief={}",
        prune,
        e0_cache_scope.as_str(),
        e0_empty_cache.as_str(),
        signature_matrix_cache.as_str(),
        allocator_relief,
    );
}

fn parse_e0_cache_scope(args: &[String]) -> Result<E0CacheScope, String> {
    let Some(raw) = parse_string_flag(args, "--e0-cache-scope")? else {
        return Ok(E0CacheScope::Profile);
    };
    match raw.as_str() {
        "profile" => Ok(E0CacheScope::Profile),
        "step" => Ok(E0CacheScope::Step),
        _ => Err(format!(
            "--e0-cache-scope expects `profile` or `step`, got `{raw}`"
        )),
    }
}

fn parse_signature_matrix_cache(args: &[String]) -> Result<SignatureMatrixCache, String> {
    let Some(raw) = parse_string_flag(args, "--matrix-cache-scope")? else {
        return Ok(SignatureMatrixCache::DropAfterUse);
    };
    match raw.as_str() {
        "keep" => Ok(SignatureMatrixCache::Keep),
        "drop" => Ok(SignatureMatrixCache::DropAfterUse),
        _ => Err(format!(
            "--matrix-cache-scope expects `keep` or `drop`, got `{raw}`"
        )),
    }
}

fn parse_e0_empty_cache_policy(args: &[String]) -> Result<E0EmptyCachePolicy, String> {
    let Some(raw) = parse_string_flag(args, "--e0-empty-cache")? else {
        return Ok(E0EmptyCachePolicy::SpeedHash);
    };
    match raw.as_str() {
        "speed" | "hash" => Ok(E0EmptyCachePolicy::SpeedHash),
        "compact" | "segments" => Ok(E0EmptyCachePolicy::CompactSegments),
        _ => Err(format!(
            "--e0-empty-cache expects `speed` or `compact`, got `{raw}`"
        )),
    }
}

fn parse_allocator_relief(args: &[String]) -> Result<bool, String> {
    let Some(raw) = parse_string_flag(args, "--allocator-relief")? else {
        return Ok(false);
    };
    match raw.as_str() {
        "off" => Ok(false),
        "on" => Ok(true),
        _ => Err(format!(
            "--allocator-relief expects `off` or `on`, got `{raw}`"
        )),
    }
}

fn parse_batch_commit_check(args: &[String]) -> Result<bool, String> {
    let Some(raw) = parse_string_flag(args, "--batch-commit-check")? else {
        return Ok(false);
    };
    match raw.as_str() {
        "on" | "true" | "yes" => Ok(true),
        "off" | "false" | "no" => Ok(false),
        _ => Err(format!(
            "--batch-commit-check expects `on` or `off`, got `{raw}`"
        )),
    }
}

fn parse_batch_scheduler(args: &[String]) -> Result<FixedTBatchScheduler, String> {
    parse_scheduler_flag_value(
        parse_string_flag(args, "--batch-scheduler")?,
        "--batch-scheduler",
    )
}

fn parse_scheduler_flag_value(
    raw: Option<String>,
    flag_label: &str,
) -> Result<FixedTBatchScheduler, String> {
    let Some(raw) = raw else {
        return Ok(FixedTBatchScheduler::WeightedContiguousBuildCost);
    };
    match raw.as_str() {
        "contiguous" | "block" | "blocks" => Ok(FixedTBatchScheduler::Contiguous),
        "round-robin" | "round_robin" | "rr" => Ok(FixedTBatchScheduler::RoundRobin),
        "weighted-contiguous" | "weighted_contiguous" | "weighted" | "balanced" => {
            Ok(FixedTBatchScheduler::WeightedContiguous)
        }
        "weighted-contiguous-build-cost"
        | "weighted_contiguous_build_cost"
        | "build-cost"
        | "build_cost"
        | "matrix-build-cost"
        | "matrix_build_cost" => Ok(FixedTBatchScheduler::WeightedContiguousBuildCost),
        "minimax-build-cost"
        | "minimax_build_cost"
        | "weighted-minimax-build-cost"
        | "weighted_minimax_build_cost"
        | "build-cost-minimax"
        | "build_cost_minimax" => Ok(FixedTBatchScheduler::WeightedMinimaxBuildCost),
        "low-prefix-build-cost"
        | "low_prefix_build_cost"
        | "front-low-build-cost"
        | "front_low_build_cost" => Ok(FixedTBatchScheduler::LowPrefixBuildCost),
        "adaptive-contiguous" | "adaptive_contiguous" | "adaptive" => {
            Ok(FixedTBatchScheduler::AdaptiveContiguous)
        }
        "adaptive-minimax-contiguous"
        | "adaptive_minimax_contiguous"
        | "adaptive-minimax"
        | "adaptive_minimax"
        | "minimax-contiguous"
        | "minimax_contiguous"
        | "minimax" => Ok(FixedTBatchScheduler::AdaptiveMinimaxContiguous),
        "adaptive-sticky-contiguous"
        | "adaptive_sticky_contiguous"
        | "adaptive-sticky"
        | "adaptive_sticky"
        | "sticky-contiguous"
        | "sticky_contiguous"
        | "sticky" => Ok(FixedTBatchScheduler::AdaptiveStickyContiguous),
        "profile-contiguous" | "profile_contiguous" | "profile" => {
            Ok(FixedTBatchScheduler::ProfileContiguous)
        }
        "weighted-greedy" | "weighted_greedy" | "greedy" | "lpt" => {
            Ok(FixedTBatchScheduler::WeightedGreedy)
        }
        "weighted-contiguous-front-merge-tail-split"
        | "weighted_contiguous_front_merge_tail_split"
        | "front-merge-tail-split"
        | "front_merge_tail_split"
        | "merge-front-split-tail"
        | "merge_front_split_tail" => {
            Ok(FixedTBatchScheduler::WeightedContiguousFrontMergeTailSplit)
        }
        _ => Err(format!(
            "{flag_label} expects `contiguous`, `round-robin`, `weighted-contiguous`, `weighted-contiguous-build-cost`, `weighted-contiguous-front-merge-tail-split`, `adaptive-contiguous`, `adaptive-minimax-contiguous`, `adaptive-sticky-contiguous`, `profile-contiguous`, or `weighted-greedy`, got `{raw}`"
        )),
    }
}

fn checkpoint_schedule_for(args: &[String]) -> Result<CheckpointSchedule, String> {
    match parse_usize_flag(args, "--checkpoint-every-layers")? {
        Some(0) => Err("--checkpoint-every-layers expects a positive integer".to_string()),
        Some(layers) => Ok(CheckpointSchedule::EveryLayers(layers)),
        None => Ok(CheckpointSchedule::FinalOnly),
    }
}

fn checkpoint_is_due(
    schedule: CheckpointSchedule,
    cursor: ComputeCursor,
    last_checkpoint_t: usize,
) -> bool {
    if cursor.next_s != 0 {
        return false;
    }
    let completed_t = cursor.next_t.saturating_sub(1);
    match schedule {
        CheckpointSchedule::EveryLayers(layers) => {
            completed_t.saturating_sub(last_checkpoint_t) >= layers
        }
        CheckpointSchedule::FinalOnly => false,
    }
}

fn configure_rayon_threads(args: &[String]) -> Result<(), String> {
    let Some(threads) = parse_usize_flag(args, "--threads")? else {
        return Ok(());
    };
    if threads == 0 {
        return Err("--threads expects a positive integer".to_string());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|e| format!("failed to configure Rayon thread pool: {e}"))?;
    eprintln!("parallel nassau_min_res threads={threads}");
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct ProcessUsage {
    user_s: f64,
    system_s: f64,
}

impl ProcessUsage {
    fn now() -> Option<Self> {
        process_usage_now()
    }

    fn saturating_since(self, earlier: Self) -> Self {
        Self {
            user_s: (self.user_s - earlier.user_s).max(0.0),
            system_s: (self.system_s - earlier.system_s).max(0.0),
        }
    }
}

#[cfg(unix)]
fn process_usage_now() -> Option<ProcessUsage> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    Some(ProcessUsage {
        user_s: timeval_seconds(usage.ru_utime),
        system_s: timeval_seconds(usage.ru_stime),
    })
}

#[cfg(unix)]
fn timeval_seconds(value: libc::timeval) -> f64 {
    value.tv_sec as f64 + value.tv_usec as f64 / 1_000_000.0
}

#[cfg(not(unix))]
fn process_usage_now() -> Option<ProcessUsage> {
    None
}

fn print_layer_status(
    started: Instant,
    progress: ComputeProgress,
    layer_elapsed: Duration,
    layer_cpu: Option<ProcessUsage>,
    verbose: bool,
) {
    if !verbose {
        eprintln!(
            "t={}/{} generators={} layer_s={:.3} elapsed_s={:.3}",
            progress.t,
            progress.max_t,
            progress.generators,
            layer_elapsed.as_secs_f64(),
            started.elapsed().as_secs_f64(),
        );
        return;
    }
    let cpu_fields = layer_cpu
        .map(|layer_cpu| {
            format!(
                " layer_user_s={:.3} layer_system_s={:.3}",
                layer_cpu.user_s, layer_cpu.system_s
            )
        })
        .unwrap_or_default();
    let memory_fields = ProcessMemory::now()
        .map(format_memory_status_fields)
        .unwrap_or_default();
    eprintln!(
        "status nassau_min_res completed_t={} completed_task_s={} layer_s={:.3}{} elapsed_s={:.3} max_t={} max_s={} generators={}{}",
        progress.t,
        progress.s,
        layer_elapsed.as_secs_f64(),
        cpu_fields,
        started.elapsed().as_secs_f64(),
        progress.max_t,
        progress.max_t,
        progress.generators,
        memory_fields,
    );
}

fn format_memory_status_fields(memory: ProcessMemory) -> String {
    let pressure_fields = if memory.compressed_bytes >= MEMORY_PRESSURE_COMPRESSED_WARN_BYTES
        || memory.compressed_peak_bytes >= MEMORY_PRESSURE_COMPRESSED_WARN_BYTES
    {
        " memory_pressure=warn memory_pressure_reason=compressed"
    } else {
        " memory_pressure=ok"
    };
    format!(
        " rss_gib={:.3} rss_peak_gib={:.3} compressed_gib={:.3} compressed_peak_gib={:.3} phys_footprint_gib={:.3} phys_footprint_peak_gib={:.3}{}",
        bytes_to_gib(memory.resident_bytes),
        bytes_to_gib(memory.resident_peak_bytes),
        bytes_to_gib(memory.compressed_bytes),
        bytes_to_gib(memory.compressed_peak_bytes),
        bytes_to_gib(memory.phys_footprint_bytes),
        bytes_to_gib_signed(memory.phys_footprint_peak_bytes),
        pressure_fields,
    )
}

fn parse_subalgebra_list(args: &[String], max_t: usize) -> Result<Vec<Subalgebra>, String> {
    let raw = parse_string_flag(args, "--subalgebras")?
        .or(parse_string_flag(args, "--subalgebra")?)
        .unwrap_or_else(|| "A3,B3321,B3221,B3211,A2,A1,A0,F2,F1".to_string());
    let mut out = Vec::new();
    for item in raw.split(',') {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(Subalgebra::parse(trimmed, max_t)?);
    }
    if out.is_empty() {
        return Err(
            "--subalgebras must contain at least one profile, e.g. A3,B3321,B3221,B3211,A2,A1,A0,F2,F1"
                .into(),
        );
    }
    Ok(out)
}

fn configure_subalgebra_selection(args: &[String]) -> Result<(), String> {
    let mode = parse_string_flag(args, "--subalgebra-selection")?
        .as_deref()
        .map(SubalgebraSelectionMode::parse)
        .transpose()?
        .unwrap_or(SubalgebraSelectionMode::Detailed);
    set_subalgebra_selection_mode(mode);
    Ok(())
}

fn multiply_cmd(args: &[String]) -> Result<(), String> {
    if has_flag(args, "-h") || has_flag(args, "--help") {
        println!("usage: nassau_min_res multiply R S");
        return Ok(());
    }
    validate_no_flags(args)?;
    if args.len() != 2 {
        return Err(
            "usage: nassau_min_res multiply R S, e.g. multiply 2 1 or multiply 3,1 1".into(),
        );
    }
    let left = Milnor::parse(&args[0])?;
    let right = Milnor::parse(&args[1])?;
    let product = milnor::multiply(&left, &right);
    if product.is_empty() {
        println!("0");
    } else {
        let terms = product
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" + ");
        println!("{left} * {right} = {terms}");
    }
    Ok(())
}

fn basis_cmd(args: &[String]) -> Result<(), String> {
    validate_flags(args, BASIS_FLAGS)?;
    if has_flag(args, "-h") || has_flag(args, "--help") {
        println!("usage: nassau_min_res basis --degree N");
        return Ok(());
    }
    let degree =
        parse_usize_flag(args, "--degree")?.ok_or_else(|| "basis needs --degree N".to_string())?;
    let basis = milnor::basis_of_degree(degree);
    println!(
        "Milnor basis in degree {degree} ({} element(s)):",
        basis.len()
    );
    for b in basis {
        println!("  {b}");
    }
    Ok(())
}

fn range_cmd(args: &[String]) -> Result<(), String> {
    validate_flags(args, RANGE_FLAGS)?;
    if has_flag(args, "-h") || has_flag(args, "--help") {
        println!(
            "usage: nassau_min_res range --s S (--t T | --stem N) --family A|F|Fprime --n N [--tau TAU]"
        );
        return Ok(());
    }
    let s = parse_usize_flag(args, "--s")?.ok_or_else(|| "range needs --s S".to_string())?;
    let t = match parse_usize_flag(args, "--t")? {
        Some(t) => t,
        None => {
            let stem = parse_usize_flag(args, "--stem")?
                .ok_or_else(|| "range needs --t T or --stem N".to_string())?;
            s + stem
        }
    };
    let family = parse_string_flag(args, "--family")?.unwrap_or_else(|| "A".to_string());
    let n = parse_usize_flag(args, "--n")?.unwrap_or(0);
    let t_i = t as i128;
    let s_i = s as i128;

    match family.as_str() {
        "A" | "a" | "below" => {
            let rho = (1_i128 << (n + 1)) - 1;
            let tau = parse_usize_flag(args, "--tau")?
                .map(|x| x as i128)
                .unwrap_or_else(|| milnor::tau_a(n) as i128);
            let bound = rho * s_i + tau;
            println!("Window-lemma below vanishing line");
            println!("B = A({n}) unless --tau overrides tau_B");
            println!("rho_n = {rho}, tau_B = {tau}");
            println!("criterion: t > rho_n * s + tau_B");
            println!("for (s,t)=({s},{t}): {t} > {bound} is {}", t_i > bound);
        }
        "F" | "f" | "above" => {
            let slope = (1_i128 << (n + 1)) - 1;
            let bound = slope * s_i;
            println!("Theorem 4.1 / above vanishing line");
            println!("criterion: t < (2^(n+1)-1) * s for B subset F(n)");
            println!(
                "for n={n}, (s,t)=({s},{t}): {t} < {bound} is {}",
                t_i < bound
            );
        }
        "Fprime" | "fprime" | "F'" | "f'" => {
            let slope = (1_i128 << (n + 1)) - 2;
            let bound = slope * s_i;
            println!("Theorem 4.2 / F'(n) variant");
            println!("criterion: t < (2^(n+1)-2) * s for B subset F'(n)");
            println!(
                "for n={n}, (s,t)=({s},{t}): {t} < {bound} is {}",
                t_i < bound
            );
        }
        _ => return Err("family must be A, F, or Fprime".into()),
    }
    Ok(())
}

fn export_cmd(args: &[String]) -> Result<(), String> {
    validate_flags(args, EXPORT_FLAGS)?;
    if has_flag(args, "-h") || has_flag(args, "--help") {
        println!("usage: nassau_min_res export --checkpoint DIR [--output FILE] [--overwrite]");
        println!(
            "Export an existing NMR_SHARDED_V1 directory to a bidegree rank CSV with columns s,t,rank."
        );
        return Ok(());
    }

    let checkpoint_path = parse_string_flag(args, "--checkpoint")?.ok_or_else(|| {
        "export needs --checkpoint DIR for an NMR_SHARDED_V1 directory".to_string()
    })?;
    let output_path = parse_string_flag_any(args, &["--output", "--out"])?
        .unwrap_or_else(|| DEFAULT_EXPORT_PATH.to_string());
    let overwrite = has_flag(args, "--overwrite");
    let output = PathBuf::from(&output_path);
    ensure_output_path_is_free(&output, overwrite)?;
    let checkpoint = Path::new(&checkpoint_path);
    if checkpoint.is_dir() {
        let manifest = sharded_io::read_manifest(checkpoint)?;
        if manifest.format_name != sharded_io::SHARDED_CHECKPOINT_FORMAT {
            return Err(format!(
                "checkpoint directory {} has format `{}`, expected `{}`",
                checkpoint.display(),
                manifest.format_name,
                sharded_io::SHARDED_CHECKPOINT_FORMAT
            ));
        }
        let rows = sharded_io::export_sharded_generators_csv(
            checkpoint,
            manifest.completed_internal_degree,
            &output,
        )?;
        println!(
            "exported {} nonzero bidegree ranks with t <= {} from {} to {}",
            rows,
            manifest.completed_internal_degree,
            checkpoint_path,
            output.display()
        );
        println!(
            "checkpoint completed full layers t <= {}",
            manifest.completed_internal_degree
        );
        return Ok(());
    }

    Err(format!(
        "checkpoint {} is not an NMR_SHARDED_V1 directory; v4 checkpoint export was removed",
        checkpoint.display()
    ))
}

fn checkpoint_cmd(args: &[String]) -> Result<(), String> {
    if args.is_empty() || has_flag(args, "-h") || has_flag(args, "--help") {
        println!("usage: nassau_min_res checkpoint <init-sharded|verify-sharded> ...");
        println!("  init-sharded --out DIR [--overwrite]");
        println!("  verify-sharded --checkpoint DIR");
        return Ok(());
    }
    let subcmd = args[0].as_str();
    let rest = &args[1..];
    match subcmd {
        "init-sharded" | "init_sharded" | "seed-sharded" | "seed_sharded" => {
            checkpoint_init_sharded_cmd(rest)
        }
        "verify-sharded" | "verify_sharded" => checkpoint_verify_sharded_cmd(rest),
        _ => Err(format!(
            "unknown checkpoint subcommand `{subcmd}`; expected init-sharded or verify-sharded"
        )),
    }
}

fn checkpoint_init_sharded_cmd(args: &[String]) -> Result<(), String> {
    validate_flags(args, CHECKPOINT_INIT_FLAGS)?;
    let out = parse_string_flag_any(args, &["--out", "--output"])?
        .ok_or_else(|| "checkpoint init-sharded needs --out DIR".to_string())?;
    let overwrite = has_flag(args, "--overwrite");
    let resolution = Resolution::new(0);
    let manifest = sharded_io::write_sharded_checkpoint(
        Path::new(&out),
        &resolution,
        ComputeCursor::start(),
        None,
        source_git_commit(),
        overwrite,
    )?;
    let report = sharded_io::verify_sharded_checkpoint(Path::new(&out))?;
    println!("sharded_checkpoint_written={out}");
    println!("format={}", manifest.format_name);
    println!(
        "completed_internal_degree={}",
        manifest.completed_internal_degree
    );
    println!("q_blocks={}", report.q_block_count);
    println!("generators={}", report.total_generator_count);
    println!(
        "differential_terms={}",
        report.total_differential_term_count
    );
    Ok(())
}

fn checkpoint_verify_sharded_cmd(args: &[String]) -> Result<(), String> {
    validate_flags(args, CHECKPOINT_VERIFY_FLAGS)?;
    let checkpoint = parse_string_flag_any(args, &["--checkpoint", "--input"])?
        .ok_or_else(|| "checkpoint verify-sharded needs --checkpoint DIR".to_string())?;
    let dir = Path::new(&checkpoint);
    let report = sharded_io::verify_sharded_checkpoint(dir)?;
    println!("checkpoint={}", dir.display());
    println!("format={}", report.format_name);
    println!(
        "completed_internal_degree={}",
        report.completed_internal_degree
    );
    println!("q_blocks={}", report.q_block_count);
    println!("max_homological_degree={}", report.max_homological_degree);
    println!("generators={}", report.total_generator_count);
    println!(
        "differential_terms={}",
        report.total_differential_term_count
    );
    Ok(())
}

fn detailed_subalgebras_cmd(args: &[String]) -> Result<(), String> {
    validate_flags(args, DETAILED_SUBALGEBRAS_FLAGS)?;
    if has_flag(args, "-h") || has_flag(args, "--help") {
        println!("usage: nassau_min_res detailed-subalgebras [--output FILE]");
        println!("Print detailed Ext_B table coverage for registered finite subalgebras.");
        return Ok(());
    }

    let rows = detailed_ext_table_coverages();
    let mut out = String::new();
    out.push_str("algebra,profile,dim,tau,state,s_min,s_max,u_min,u_max,nonzero_entries,bitset_path,metadata_path,error\n");
    for row in rows {
        let profile = row
            .profile
            .iter()
            .map(|entry| entry.to_string())
            .collect::<Vec<_>>()
            .join(";");
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            row.algebra_name,
            profile,
            row.dim,
            row.tau,
            row.state,
            row.s_min.map(|value| value.to_string()).unwrap_or_default(),
            row.s_max.map(|value| value.to_string()).unwrap_or_default(),
            row.u_min.map(|value| value.to_string()).unwrap_or_default(),
            row.u_max.map(|value| value.to_string()).unwrap_or_default(),
            row.nonzero_entries
                .map(|value| value.to_string())
                .unwrap_or_default(),
            row.bitset_path,
            row.metadata_path,
            row.error.unwrap_or_default().replace(',', ";"),
        ));
    }

    if let Some(output) = parse_string_flag_any(args, &["--output", "--out"])?.map(PathBuf::from) {
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create detailed subalgebra output directory {}: {e}",
                    parent.display()
                )
            })?;
        }
        std::fs::write(&output, out).map_err(|e| {
            format!(
                "failed to write detailed subalgebra coverage {}: {e}",
                output.display()
            )
        })?;
        println!("wrote detailed subalgebra coverage to {}", output.display());
    } else {
        print!("{out}");
    }
    Ok(())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn parse_usize_flag(args: &[String], flag: &str) -> Result<Option<usize>, String> {
    parse_string_flag(args, flag)?
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("{flag} expects a non-negative integer, got `{value}`"))
        })
        .transpose()
}

fn parse_string_flag(args: &[String], flag: &str) -> Result<Option<String>, String> {
    parse_string_flag_any(args, &[flag])
}

fn parse_string_flag_any(args: &[String], flags: &[&str]) -> Result<Option<String>, String> {
    let mut found = None;
    for (i, arg) in args.iter().enumerate() {
        if flags.iter().any(|flag| arg == flag) {
            let value = args
                .get(i + 1)
                .ok_or_else(|| format!("{} needs a value", flags.join("/")))?;
            found = Some(value.clone());
        } else {
            for flag in flags {
                let prefix = format!("{flag}=");
                if let Some(value) = arg.strip_prefix(&prefix) {
                    found = Some(value.to_string());
                }
            }
        }
    }
    Ok(found)
}

fn ensure_output_path_is_free(path: &Path, overwrite: bool) -> Result<(), String> {
    if overwrite || !path.exists() {
        return Ok(());
    }
    Err(format!(
        "refusing to overwrite existing output {}; pass --overwrite to replace it or choose a different --output",
        path.display()
    ))
}

#[derive(Clone, Copy)]
enum FlagKind {
    Bool,
    Value,
}

const COMPUTE_FLAGS: &[(&str, FlagKind)] = &[
    ("--t", FlagKind::Value),
    ("--show-differentials", FlagKind::Bool),
    ("--show-steps", FlagKind::Bool),
    ("--print-report", FlagKind::Bool),
    ("--verbose", FlagKind::Bool),
    ("--output", FlagKind::Value),
    ("--out", FlagKind::Value),
    ("--checkpoint", FlagKind::Value),
    ("--load-checkpoint", FlagKind::Value),
    ("--save-checkpoint", FlagKind::Value),
    ("--checkpoint-every-layers", FlagKind::Value),
    ("--cache-prune-min-t", FlagKind::Value),
    ("--cache-window-t", FlagKind::Value),
    ("--e0-cache-scope", FlagKind::Value),
    ("--e0-empty-cache", FlagKind::Value),
    ("--matrix-cache-scope", FlagKind::Value),
    ("--allocator-relief", FlagKind::Value),
    ("--fresh", FlagKind::Bool),
    ("--no-checkpoint", FlagKind::Bool),
    ("--threads", FlagKind::Value),
    ("--batch-workers", FlagKind::Value),
    ("--batch-scheduler", FlagKind::Value),
    ("--batch-inner", FlagKind::Value),
    ("--batch-commit-check", FlagKind::Value),
    ("--allow-full-naive-batch", FlagKind::Bool),
    ("--algorithm", FlagKind::Value),
    ("--subalgebras", FlagKind::Value),
    ("--subalgebra", FlagKind::Value),
    ("--subalgebra-selection", FlagKind::Value),
    ("--strict", FlagKind::Bool),
    ("--force", FlagKind::Bool),
    ("-h", FlagKind::Bool),
    ("--help", FlagKind::Bool),
];

const EXPORT_FLAGS: &[(&str, FlagKind)] = &[
    ("--checkpoint", FlagKind::Value),
    ("--output", FlagKind::Value),
    ("--out", FlagKind::Value),
    ("--overwrite", FlagKind::Bool),
    ("-h", FlagKind::Bool),
    ("--help", FlagKind::Bool),
];

const CHECKPOINT_INIT_FLAGS: &[(&str, FlagKind)] = &[
    ("--output", FlagKind::Value),
    ("--out", FlagKind::Value),
    ("--overwrite", FlagKind::Bool),
    ("-h", FlagKind::Bool),
    ("--help", FlagKind::Bool),
];

const CHECKPOINT_VERIFY_FLAGS: &[(&str, FlagKind)] = &[
    ("--input", FlagKind::Value),
    ("--checkpoint", FlagKind::Value),
    ("-h", FlagKind::Bool),
    ("--help", FlagKind::Bool),
];

const DETAILED_SUBALGEBRAS_FLAGS: &[(&str, FlagKind)] = &[
    ("--output", FlagKind::Value),
    ("--out", FlagKind::Value),
    ("-h", FlagKind::Bool),
    ("--help", FlagKind::Bool),
];

const BASIS_FLAGS: &[(&str, FlagKind)] = &[
    ("--degree", FlagKind::Value),
    ("-h", FlagKind::Bool),
    ("--help", FlagKind::Bool),
];

const RANGE_FLAGS: &[(&str, FlagKind)] = &[
    ("--s", FlagKind::Value),
    ("--t", FlagKind::Value),
    ("--stem", FlagKind::Value),
    ("--family", FlagKind::Value),
    ("--n", FlagKind::Value),
    ("--tau", FlagKind::Value),
    ("-h", FlagKind::Bool),
    ("--help", FlagKind::Bool),
];

fn validate_flags(args: &[String], specs: &[(&str, FlagKind)]) -> Result<(), String> {
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with('-') {
            return Err(format!("unexpected argument `{arg}`"));
        }
        let (name, inline_value) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value)))
            .unwrap_or((arg.as_str(), None));
        let Some((_, kind)) = specs.iter().find(|(flag, _)| *flag == name) else {
            return Err(format!("unknown option `{name}`; run with --help"));
        };
        match kind {
            FlagKind::Bool => {
                if inline_value.is_some() {
                    return Err(format!("{name} does not take a value"));
                }
            }
            FlagKind::Value => {
                if inline_value.is_none() {
                    let Some(value) = args.get(i + 1) else {
                        return Err(format!("{name} needs a value"));
                    };
                    if value.starts_with('-') {
                        return Err(format!("{name} needs a value, got option `{value}`"));
                    }
                    i += 1;
                }
            }
        }
        i += 1;
    }
    Ok(())
}

fn validate_no_flags(args: &[String]) -> Result<(), String> {
    for arg in args {
        if arg.starts_with('-') {
            return Err(format!("unexpected option `{arg}`; run with --help"));
        }
    }
    Ok(())
}

fn print_help() {
    println!(
        "\
nassau_min_res

Computes a low-dimensional minimal free resolution of F2 over the mod-2
Steenrod algebra, using the Milnor basis.

Commands:
  compute --t N [--threads N] [--checkpoint DIR]
      Compute through internal degree t <= N. The default fixed-t batch mode
      uses automatic finite-subalgebra selection. Run `compute --help` for all
      compute options.

  checkpoint init-sharded --out DIR [--overwrite]
      Create a seed NMR_SHARDED_V1 checkpoint.

  checkpoint verify-sharded --checkpoint DIR
      Verify checkpoint structure and generator references.

  export --checkpoint DIR [--output FILE] [--overwrite]
      Export one row per nonzero (s,t) to CSV, with columns s,t,rank.

  detailed-subalgebras [--output FILE]
      Print coverage information for the bundled detailed Ext_B tables.

  multiply R S
      Multiply two Milnor basis elements.

  basis --degree N
      Print the Milnor basis in internal degree N.

  range --s S (--t T | --stem N) --family A|F|Fprime --n N [--tau TAU]
      Check the vanishing-line criteria used by the accelerated algorithm.
"
    );
}

fn print_compute_help() {
    println!(
        "\
usage: nassau_min_res compute [OPTIONS]

Core options:
  --t N                        Maximum internal degree (default: 12)
  --threads N                  Number of Rayon worker threads
  --checkpoint DIR             Load from and save to the same checkpoint
  --load-checkpoint DIR        Read an existing checkpoint
  --save-checkpoint DIR        Write a separate checkpoint
  --no-checkpoint              Do not read or write a checkpoint
  --fresh                      Ignore an existing checkpoint and start over
  --output FILE, --out FILE    Write a human-readable report
  --show-differentials         Include differentials in the report
  --show-steps                 Print individual calculation steps

Algorithm options:
  --algorithm NAME             fixed-t-batch, auto, nassau, naive, or
                               fixed-t-batch-shadow (default: fixed-t-batch)
  --subalgebras LIST           Comma-separated priority list for auto mode
  --subalgebra NAME            Finite subalgebra for nassau mode
  --subalgebra-selection MODE  detailed or original
  --strict                     Disable the direct fallback where applicable
  --force                      Force the requested accelerated path

Checkpoint and memory tuning:
  --checkpoint-every-layers N  Save after every N completed t layers
                               (default: save only at the end; use 1 for every layer)
  --cache-prune-min-t N
  --cache-window-t N
  --e0-cache-scope VALUE
  --e0-empty-cache VALUE
  --matrix-cache-scope VALUE
  --allocator-relief off|on

Fixed-t batch tuning:
  --batch-workers N            Outer groups (default: half the Rayon threads,
                               capped at 4)
  --batch-scheduler VALUE
  --batch-inner auto|nassau|naive
  --batch-commit-check on|off   Extra per-layer validation (default: off)
  --allow-full-naive-batch

Other:
  --print-report
  --verbose                    Print detailed runtime diagnostics
  -h, --help
"
    );
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn compute_mode_defaults_to_fixed_t_batch_with_adaptive_workers() {
        let mode = build_compute_mode(&[], 150).unwrap();
        match mode {
            ComputeMode::FixedTBatch {
                workers,
                shadow,
                validate_commit,
                scheduler,
                inner,
            } => {
                assert_eq!(workers, Some(default_fixed_t_batch_workers()));
                assert!(!shadow);
                assert!(!validate_commit);
                assert_eq!(scheduler, FixedTBatchScheduler::WeightedContiguousBuildCost);
                assert!(matches!(
                    *inner,
                    ComputeMode::AutoBoundedNaiveFallback { .. }
                ));
            }
            _ => panic!("default compute mode was not fixed-t-batch"),
        }
    }

    #[test]
    fn default_fixed_t_workers_are_half_the_threads_capped_at_four() {
        let cases = [
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 2),
            (6, 3),
            (8, 4),
            (10, 4),
            (16, 4),
            (64, 4),
        ];
        for (threads, expected) in cases {
            assert_eq!(
                default_fixed_t_batch_workers_for_threads(threads),
                expected,
                "unexpected default for {threads} Rayon threads"
            );
        }
    }

    #[test]
    fn explicit_auto_mode_remains_available() {
        let mode = build_compute_mode(&args(&["--algorithm", "auto"]), 150).unwrap();
        assert!(matches!(mode, ComputeMode::Auto { .. }));
    }

    #[test]
    fn compute_range_accepts_only_t() {
        assert!(validate_flags(&args(&["--t", "20"]), COMPUTE_FLAGS).is_ok());
        assert!(validate_flags(&args(&["--max-t", "20"]), COMPUTE_FLAGS).is_err());
        assert!(validate_flags(&args(&["--max-s", "10"]), COMPUTE_FLAGS).is_err());
    }

    #[test]
    fn checkpoints_are_final_only_by_default() {
        assert!(matches!(
            checkpoint_schedule_for(&[]).unwrap(),
            CheckpointSchedule::FinalOnly
        ));
    }

    #[test]
    fn checkpoint_layer_interval_enables_intermediate_saves() {
        let schedule =
            checkpoint_schedule_for(&args(&["--checkpoint-every-layers", "10"])).unwrap();
        assert!(matches!(schedule, CheckpointSchedule::EveryLayers(10)));
    }

    #[test]
    fn checkpoint_layer_interval_is_relative_to_the_last_save() {
        let schedule = CheckpointSchedule::EveryLayers(10);
        assert!(!checkpoint_is_due(
            schedule,
            ComputeCursor {
                next_t: 110,
                next_s: 0
            },
            100
        ));
        assert!(checkpoint_is_due(
            schedule,
            ComputeCursor {
                next_t: 111,
                next_s: 0
            },
            100
        ));
    }

    #[test]
    fn checkpoint_every_layer_uses_one() {
        assert!(checkpoint_is_due(
            CheckpointSchedule::EveryLayers(1),
            ComputeCursor {
                next_t: 101,
                next_s: 0
            },
            99
        ));
    }

    #[test]
    fn checkpoint_layer_interval_must_be_positive() {
        assert!(checkpoint_schedule_for(&args(&["--checkpoint-every-layers", "0"])).is_err());
    }
}
