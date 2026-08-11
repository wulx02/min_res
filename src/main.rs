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

const DEFAULT_CHECKPOINT_DIR: &str = "checkpoint";
const DEFAULT_RANK_OUTPUT_DIR: &str = "rank_output";
const DEFAULT_RESOLUTION_OUTPUT_DIR: &str = "resolution_output";
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
        "export-bidegree-ranks" => export_bidegree_ranks_cmd(&args),
        "export-resolution" => export_resolution_cmd(&args),
        "detailed-subalgebras" => detailed_subalgebras_cmd(&args),
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
    let checkpoint_paths = checkpoint_paths_for(args, max_t)?;
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
        checkpoint_paths.auto_discover_load,
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
    let mut last_auto_saved_checkpoint = None;
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
                save_checkpoint_for_cursor(
                    &checkpoint_paths,
                    res,
                    cursor,
                    verbose,
                    &mut last_auto_saved_checkpoint,
                )?;
                last_checkpoint_t = cursor.next_t.saturating_sub(1);
            }
            if completed_layer {
                layer_t = cursor.next_t;
                layer_timer = Instant::now();
                layer_usage = ProcessUsage::now();
            }
            Ok(())
        })?;
    save_checkpoint_for_cursor(
        &checkpoint_paths,
        &res,
        final_cursor,
        verbose,
        &mut last_auto_saved_checkpoint,
    )?;
    let compute_s = compute_timer.elapsed().as_secs_f64();

    println!(
        "computed t <= {max_t}, s <= {}, generators={}",
        max_t,
        res.generator_count()
    );
    if verbose {
        eprintln!(
            "timing ext compute_s={compute_s:.6} total_s={:.6}",
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
        "auto" | "sequential" | "algorithm2-auto" | "alg2-auto" => {
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
        "accelerated" | "algorithm2" | "alg2" => {
            let subalgebra = parse_string_flag(args, "--subalgebra")?
                .ok_or_else(|| {
                    "Algorithm 2 needs an explicit --subalgebra A0/A1/A2/...".to_string()
                })
                .and_then(|name| Subalgebra::parse(&name, max_t))?;
            let strict = has_flag(args, "--strict");
            let force = has_flag(args, "--force");
            Ok(ComputeMode::Accelerated {
                subalgebra,
                strict,
                force,
            })
        }
        "naive" | "algorithm1" | "alg1" => Ok(ComputeMode::Naive),
        _ => Err(
            "--algorithm must be sequential, auto, accelerated, naive, fixed-t-batch, or fixed-t-batch-shadow"
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
        "auto" | "sequential" | "algorithm2-auto" | "alg2-auto" => {
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
        "accelerated" | "algorithm2" | "alg2" => {
            let subalgebra = parse_string_flag(args, "--subalgebra")?
                .ok_or_else(|| {
                    "fixed-t-batch --batch-inner accelerated needs --subalgebra A0/A1/A2/..."
                        .to_string()
                })
                .and_then(|name| Subalgebra::parse(&name, max_t))?;
            let strict = has_flag(args, "--strict") || max_t > FULL_NAIVE_BATCH_DIAGNOSTIC_MAX_T;
            let force = has_flag(args, "--force");
            Ok(ComputeMode::Accelerated {
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
                    "--batch-inner naive is full frozen homology and is blocked for t > {FULL_NAIVE_BATCH_DIAGNOSTIC_MAX_T}; fixed-t batch experiments must use Algorithm 2/signature reduction via --batch-inner auto or accelerated."
                ));
            }
            Ok(ComputeMode::Naive)
        }
        _ => Err(format!(
            "--batch-inner must be auto, accelerated, or naive, got `{inner}`"
        )),
    }
}

struct CheckpointPaths {
    load: Option<PathBuf>,
    save: Option<PathBuf>,
    load_required: bool,
    auto_discover_load: bool,
    save_by_completed_t: bool,
}

fn checkpoint_paths_for(args: &[String], max_t: usize) -> Result<CheckpointPaths, String> {
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
            auto_discover_load: false,
            save_by_completed_t: false,
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
    let default_checkpoint = default_checkpoint_path(max_t);

    let explicit_load = load_checkpoint.is_some();
    let auto_discover_load = !explicit_load && legacy_checkpoint.is_none();
    let save_by_completed_t = save_checkpoint.is_none() && legacy_checkpoint.is_none();
    let load_required = explicit_load;
    let load = load_checkpoint
        .or_else(|| legacy_checkpoint.clone())
        .or_else(|| Some(default_checkpoint.clone()));
    let save = save_checkpoint
        .or_else(|| legacy_checkpoint.clone())
        .or_else(|| Some(default_checkpoint.clone()));

    Ok(CheckpointPaths {
        load,
        save,
        load_required,
        auto_discover_load,
        save_by_completed_t,
    })
}

fn default_checkpoint_path(max_t: usize) -> PathBuf {
    Path::new(DEFAULT_CHECKPOINT_DIR).join(format!("t{max_t}.checkpoint"))
}

fn default_rank_output_path(max_t: usize) -> PathBuf {
    Path::new(DEFAULT_RANK_OUTPUT_DIR).join(format!("t{max_t}.csv"))
}

fn default_resolution_output_path(max_t: usize) -> PathBuf {
    Path::new(DEFAULT_RESOLUTION_OUTPUT_DIR).join(format!("t{max_t}.jsonl"))
}

fn checkpoint_degree_from_name(name: &str) -> Option<usize> {
    name.strip_prefix('t')?
        .strip_suffix(".checkpoint")?
        .parse()
        .ok()
}

fn latest_checkpoint_in_dir(dir: &Path, max_t: usize) -> Result<Option<PathBuf>, String> {
    if !dir.exists() {
        return Ok(None);
    }
    if !dir.is_dir() {
        return Err(format!(
            "default checkpoint location {} exists but is not a directory",
            dir.display()
        ));
    }

    let mut latest: Option<(usize, PathBuf)> = None;
    for entry in std::fs::read_dir(dir).map_err(|err| {
        format!(
            "failed to read checkpoint directory {}: {err}",
            dir.display()
        )
    })? {
        let entry = entry.map_err(|err| {
            format!(
                "failed to read an entry in checkpoint directory {}: {err}",
                dir.display()
            )
        })?;
        let file_type = entry.file_type().map_err(|err| {
            format!(
                "failed to inspect checkpoint candidate {}: {err}",
                entry.path().display()
            )
        })?;
        if !file_type.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(t) = checkpoint_degree_from_name(&name) else {
            continue;
        };
        if t <= max_t && latest.as_ref().is_none_or(|(latest_t, _)| t > *latest_t) {
            latest = Some((t, entry.path()));
        }
    }
    Ok(latest.map(|(_, path)| path))
}

fn latest_default_checkpoint(max_t: usize) -> Result<Option<PathBuf>, String> {
    latest_checkpoint_in_dir(Path::new(DEFAULT_CHECKPOINT_DIR), max_t)
}

fn checkpoint_display(path: Option<&PathBuf>) -> String {
    path.map(|path| path.display().to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn print_checkpoint_paths(load_path: Option<&PathBuf>, save_path: Option<&PathBuf>) {
    if load_path != save_path {
        eprintln!(
            "checkpoint ext paths load_checkpoint={} save_checkpoint={}",
            checkpoint_display(load_path),
            checkpoint_display(save_path),
        );
    }
}

fn print_startup_info(
    max_t: usize,
    load_checkpoint: Option<&PathBuf>,
    save_checkpoint: Option<&PathBuf>,
    rayon_threads: usize,
) {
    let argv = env::args().collect::<Vec<_>>();
    let cwd = env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    eprintln!(
        "startup ext package={} version={} git_commit={} cwd={} argv={:?}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        option_env!("GIT_COMMIT").unwrap_or("unknown"),
        cwd,
        argv,
    );
    eprintln!(
        "startup ext max_t={} max_s={} load_checkpoint={} save_checkpoint={} rayon_threads={} env_RAYON_NUM_THREADS={} env_EXT_FULL_DIFFERENTIAL_CHUNK_MB={}",
        max_t,
        max_t,
        checkpoint_display(load_checkpoint),
        checkpoint_display(save_checkpoint),
        rayon_threads,
        env_value("RAYON_NUM_THREADS"),
        env_value("EXT_FULL_DIFFERENTIAL_CHUNK_MB"),
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
    auto_discover_load: bool,
    fresh: bool,
    verbose: bool,
) -> Result<(Resolution, ComputeCursor), String> {
    let Some(path) = checkpoint_path else {
        if verbose {
            eprintln!("checkpoint ext start=fresh checkpoint=none reason=no_checkpoint_path");
        }
        return Ok((Resolution::new(max_t), ComputeCursor::start()));
    };

    if fresh {
        if verbose {
            eprintln!(
                "checkpoint ext start=fresh checkpoint={} reason=fresh_requested",
                path.display()
            );
        }
        return Ok((Resolution::new(max_t), ComputeCursor::start()));
    }

    let load_path = if path.exists() {
        Some(path.clone())
    } else if auto_discover_load {
        latest_default_checkpoint(max_t)?
    } else {
        None
    };

    if let Some(load_path) = load_path {
        let (resolution, cursor) = load_sharded_checkpoint_for_compute(&load_path, max_t)?;
        if cursor.is_complete_for(max_t) {
            if verbose {
                eprintln!(
                    "checkpoint ext start=complete checkpoint={} next_t={} next_s={} generators={}",
                    load_path.display(),
                    cursor.next_t,
                    cursor.next_s,
                    resolution.generator_count()
                );
            } else {
                eprintln!(
                    "loaded checkpoint {} through t={}",
                    load_path.display(),
                    cursor.next_t.saturating_sub(1)
                );
            }
        } else {
            if verbose {
                eprintln!(
                    "checkpoint ext start=resume checkpoint={} next_t={} next_s={} generators={}",
                    load_path.display(),
                    cursor.next_t,
                    cursor.next_s,
                    resolution.generator_count()
                );
            } else {
                eprintln!("resuming {} at t={}", load_path.display(), cursor.next_t);
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
                "checkpoint ext start=fresh checkpoint={} reason=no_checkpoint",
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
        return Err(format!("checkpoint {} is not a directory", path.display()));
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

fn checkpoint_save_path(
    checkpoint_paths: &CheckpointPaths,
    cursor: ComputeCursor,
) -> Option<PathBuf> {
    checkpoint_paths.save.as_ref().map(|configured_path| {
        if checkpoint_paths.save_by_completed_t {
            default_checkpoint_path(cursor.next_t.saturating_sub(1))
        } else {
            configured_path.clone()
        }
    })
}

fn save_checkpoint_for_cursor(
    checkpoint_paths: &CheckpointPaths,
    resolution: &Resolution,
    cursor: ComputeCursor,
    verbose: bool,
    last_auto_saved_checkpoint: &mut Option<PathBuf>,
) -> Result<(), String> {
    let checkpoint_path = checkpoint_save_path(checkpoint_paths, cursor);
    if let Some(path) = checkpoint_path.as_ref() {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create checkpoint directory {}: {err}",
                    parent.display()
                )
            })?;
        }
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
                "checkpoint ext saved checkpoint={} format={} elapsed_s={:.3} size_mb={:.1} next_t={} next_s={}",
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

        if checkpoint_paths.save_by_completed_t {
            if let Some(previous) = last_auto_saved_checkpoint.as_ref()
                && previous != path
            {
                sharded_io::remove_verified_checkpoint(previous).map_err(|error| {
                    format!(
                        "saved checkpoint {}, but failed to remove superseded intermediate checkpoint {}: {error}",
                        path.display(),
                        previous.display()
                    )
                })?;
                if verbose {
                    eprintln!(
                        "checkpoint ext removed_superseded checkpoint={}",
                        previous.display()
                    );
                }
            }
            *last_auto_saved_checkpoint = Some(path.clone());
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
        "cache ext invalidation=precise solver_cache=true step_solver_cache=true full_basis_lookup_cache=true algebra_basis_index_cache=true full_differential=chunked cache_pruning={} e0_cache_scope={} e0_empty_cache={} signature_matrix_cache={} allocator_relief={}",
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
    eprintln!("parallel ext threads={threads}");
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
        "status ext completed_t={} completed_task_s={} layer_s={:.3}{} elapsed_s={:.3} max_t={} max_s={} generators={}{}",
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
        println!("usage: ext multiply R S");
        return Ok(());
    }
    validate_no_flags(args)?;
    if args.len() != 2 {
        return Err("usage: ext multiply R S, e.g. multiply 2 1 or multiply 3,1 1".into());
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
        println!("usage: ext basis --degree N");
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
        println!("usage: ext range --s S (--t T | --stem N) --family A|F|Fprime --n N [--tau TAU]");
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

fn export_bidegree_ranks_cmd(args: &[String]) -> Result<(), String> {
    validate_flags(args, EXPORT_FLAGS)?;
    if has_flag(args, "-h") || has_flag(args, "--help") {
        println!(
            "usage: ext export-bidegree-ranks [--t N] [--checkpoint DIR] [--output FILE] [--overwrite]"
        );
        println!(
            "Export an existing checkpoint directory to a bidegree rank CSV with columns s,t,rank."
        );
        println!("By default, --t N reads checkpoint/tN.checkpoint and writes rank_output/tN.csv.");
        println!("With neither --t nor --checkpoint, the latest default checkpoint is exported.");
        return Ok(());
    }

    let source = checkpoint_export_source(args)?;
    let output = parse_string_flag(args, "--output")?
        .map(PathBuf::from)
        .unwrap_or_else(|| default_rank_output_path(source.export_t));
    ensure_output_path_is_free(&output, has_flag(args, "--overwrite"))?;
    let rows =
        sharded_io::export_sharded_generators_csv(&source.checkpoint, source.export_t, &output)?;
    println!(
        "exported {} nonzero bidegree ranks with t <= {} from {} to {}",
        rows,
        source.export_t,
        source.checkpoint.display(),
        output.display()
    );
    println!(
        "checkpoint completed full layers t <= {}",
        source.manifest.completed_internal_degree
    );
    Ok(())
}

fn export_resolution_cmd(args: &[String]) -> Result<(), String> {
    validate_flags(args, EXPORT_FLAGS)?;
    if has_flag(args, "-h") || has_flag(args, "--help") {
        println!(
            "usage: ext export-resolution [--t N] [--checkpoint DIR] [--output FILE] [--overwrite]"
        );
        println!(
            "Export the complete minimal resolution to versioned JSON Lines, including generators and differentials."
        );
        println!(
            "By default, --t N reads checkpoint/tN.checkpoint and writes resolution_output/tN.jsonl."
        );
        println!("With neither --t nor --checkpoint, the latest default checkpoint is exported.");
        return Ok(());
    }

    let source = checkpoint_export_source(args)?;
    let output = parse_string_flag(args, "--output")?
        .map(PathBuf::from)
        .unwrap_or_else(|| default_resolution_output_path(source.export_t));
    ensure_output_path_is_free(&output, has_flag(args, "--overwrite"))?;
    let summary =
        sharded_io::export_minimal_resolution_jsonl(&source.checkpoint, source.export_t, &output)?;
    println!(
        "exported {} generators and {} differential terms with t <= {} from {} to {}",
        summary.generator_count,
        summary.differential_term_count,
        source.export_t,
        source.checkpoint.display(),
        output.display()
    );
    println!(
        "format={} version={}",
        sharded_io::MINIMAL_RESOLUTION_JSONL_FORMAT,
        sharded_io::MINIMAL_RESOLUTION_JSONL_VERSION,
    );
    Ok(())
}

struct CheckpointExportSource {
    checkpoint: PathBuf,
    manifest: sharded_io::ShardedManifest,
    export_t: usize,
}

fn checkpoint_export_source(args: &[String]) -> Result<CheckpointExportSource, String> {
    let requested_t = parse_usize_flag(args, "--t")?;
    let checkpoint = if let Some(path) = parse_string_flag(args, "--checkpoint")? {
        PathBuf::from(path)
    } else if let Some(max_t) = requested_t {
        default_checkpoint_path(max_t)
    } else {
        latest_default_checkpoint(usize::MAX)?.ok_or_else(|| {
            format!(
                "no default checkpoints found in {}; run compute first or pass --checkpoint DIR",
                DEFAULT_CHECKPOINT_DIR
            )
        })?
    };
    if !checkpoint.is_dir() {
        return Err(format!(
            "checkpoint {} is not a supported checkpoint directory",
            checkpoint.display()
        ));
    }

    let manifest = sharded_io::read_manifest(&checkpoint)?;
    if manifest.format_name != sharded_io::SHARDED_CHECKPOINT_FORMAT {
        return Err(format!(
            "checkpoint directory {} has format `{}`, expected `{}`",
            checkpoint.display(),
            manifest.format_name,
            sharded_io::SHARDED_CHECKPOINT_FORMAT
        ));
    }
    let export_t = requested_t.unwrap_or(manifest.completed_internal_degree);
    if export_t > manifest.completed_internal_degree {
        return Err(format!(
            "checkpoint {} is complete only through t={}, so it cannot export --t {}",
            checkpoint.display(),
            manifest.completed_internal_degree,
            export_t
        ));
    }
    Ok(CheckpointExportSource {
        checkpoint,
        manifest,
        export_t,
    })
}

fn checkpoint_cmd(args: &[String]) -> Result<(), String> {
    if args.is_empty() || has_flag(args, "-h") || has_flag(args, "--help") {
        println!("usage: ext checkpoint <init|verify> ...");
        println!("  init --out DIR [--overwrite]");
        println!("  verify --checkpoint DIR");
        return Ok(());
    }
    let subcmd = args[0].as_str();
    let rest = &args[1..];
    match subcmd {
        "init" => checkpoint_init_cmd(rest),
        "verify" => checkpoint_verify_cmd(rest),
        _ => Err(format!(
            "unknown checkpoint subcommand `{subcmd}`; expected init or verify"
        )),
    }
}

fn checkpoint_init_cmd(args: &[String]) -> Result<(), String> {
    validate_flags(args, CHECKPOINT_INIT_FLAGS)?;
    let out = parse_string_flag(args, "--out")?
        .ok_or_else(|| "checkpoint init needs --out DIR".to_string())?;
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
    println!("checkpoint_written={out}");
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

fn checkpoint_verify_cmd(args: &[String]) -> Result<(), String> {
    validate_flags(args, CHECKPOINT_VERIFY_FLAGS)?;
    let checkpoint = parse_string_flag(args, "--checkpoint")?
        .ok_or_else(|| "checkpoint verify needs --checkpoint DIR".to_string())?;
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
        println!("usage: ext detailed-subalgebras [--output FILE]");
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

    if let Some(output) = parse_string_flag(args, "--output")?.map(PathBuf::from) {
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
    let mut found = None;
    for (i, arg) in args.iter().enumerate() {
        if arg == flag {
            let value = args
                .get(i + 1)
                .ok_or_else(|| format!("{flag} needs a value"))?;
            found = Some(value.clone());
        } else if let Some(value) = arg.strip_prefix(&format!("{flag}=")) {
            found = Some(value.to_string());
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
    ("--verbose", FlagKind::Bool),
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
    ("--t", FlagKind::Value),
    ("--checkpoint", FlagKind::Value),
    ("--output", FlagKind::Value),
    ("--overwrite", FlagKind::Bool),
    ("-h", FlagKind::Bool),
    ("--help", FlagKind::Bool),
];

const CHECKPOINT_INIT_FLAGS: &[(&str, FlagKind)] = &[
    ("--out", FlagKind::Value),
    ("--overwrite", FlagKind::Bool),
    ("-h", FlagKind::Bool),
    ("--help", FlagKind::Bool),
];

const CHECKPOINT_VERIFY_FLAGS: &[(&str, FlagKind)] = &[
    ("--checkpoint", FlagKind::Value),
    ("-h", FlagKind::Bool),
    ("--help", FlagKind::Bool),
];

const DETAILED_SUBALGEBRAS_FLAGS: &[(&str, FlagKind)] = &[
    ("--output", FlagKind::Value),
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
ext

Computes a low-dimensional minimal free resolution of F2 over the mod-2
Steenrod algebra, using the Milnor basis.

Commands:
  compute --t N [--threads N] [--checkpoint DIR]
      Compute through internal degree t <= N. The default fixed-t batch mode
      uses automatic finite-subalgebra selection. Run `compute --help` for all
      compute options.

  checkpoint init --out DIR [--overwrite]
      Create a seed checkpoint.

  checkpoint verify --checkpoint DIR
      Verify checkpoint structure and generator references.

  export-bidegree-ranks [--t N] [--checkpoint DIR] [--output FILE] [--overwrite]
      Export one row per nonzero (s,t) to CSV, with columns s,t,rank.

  export-resolution [--t N] [--checkpoint DIR] [--output FILE] [--overwrite]
      Export all generators and differentials to versioned JSON Lines.

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
usage: ext compute [OPTIONS]

Core options:
  --t N                        Maximum internal degree (default: 12)
  --threads N                  Number of Rayon worker threads
  --checkpoint DIR             Load from and save to the same checkpoint
  --load-checkpoint DIR        Read an existing checkpoint
  --save-checkpoint DIR        Write a separate checkpoint
  --no-checkpoint              Do not read or write a checkpoint
  --fresh                      Ignore an existing checkpoint and start over

Default output paths:
  checkpoint/tN.checkpoint     Checkpoint for a computation through --t N
                               (automatically resumes the latest earlier one)

Algorithm options:
  --algorithm NAME             fixed-t-batch, auto, accelerated, naive, or
                               fixed-t-batch-shadow (default: fixed-t-batch)
  --subalgebras LIST           Comma-separated priority list for auto mode
  --subalgebra NAME            Finite subalgebra for accelerated mode
  --subalgebra-selection MODE  detailed or original
  --strict                     Disable the direct fallback where applicable
  --force                      Force the requested accelerated path

Checkpoint and memory tuning:
  --checkpoint-every-layers N  Save after every N completed t layers
                               using the actual completed t in the default path;
                               only the newest intermediate from this run is kept
                               (default: final only; use 1 for every layer)
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
  --batch-inner auto|accelerated|naive
  --batch-commit-check on|off   Extra per-layer validation (default: off)
  --allow-full-naive-batch

Other:
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
    fn compute_rejects_removed_plain_text_report_options() {
        for option in [
            "--output",
            "--out",
            "--show-differentials",
            "--show-steps",
            "--print-report",
        ] {
            assert!(validate_flags(&args(&[option]), COMPUTE_FLAGS).is_err());
        }
    }

    #[test]
    fn public_commands_reject_removed_flag_aliases() {
        assert!(validate_flags(&args(&["--out", "file"]), EXPORT_FLAGS).is_err());
        assert!(validate_flags(&args(&["--output", "dir"]), CHECKPOINT_INIT_FLAGS).is_err());
        assert!(validate_flags(&args(&["--input", "dir"]), CHECKPOINT_VERIFY_FLAGS).is_err());
        assert!(validate_flags(&args(&["--out", "file"]), DETAILED_SUBALGEBRAS_FLAGS).is_err());
    }

    #[test]
    fn default_export_paths_distinguish_ranks_and_resolution() {
        assert_eq!(
            default_rank_output_path(100),
            PathBuf::from("rank_output/t100.csv")
        );
        assert_eq!(
            default_resolution_output_path(100),
            PathBuf::from("resolution_output/t100.jsonl")
        );
    }

    #[test]
    fn checkpoints_are_final_only_by_default() {
        assert!(matches!(
            checkpoint_schedule_for(&[]).unwrap(),
            CheckpointSchedule::FinalOnly
        ));
    }

    #[test]
    fn default_checkpoint_paths_are_named_by_target_t() {
        let paths = checkpoint_paths_for(&[], 100).unwrap();
        let expected = Some(PathBuf::from("checkpoint/t100.checkpoint"));
        assert_eq!(paths.load, expected);
        assert_eq!(paths.save, expected);
        assert!(paths.auto_discover_load);
        assert!(paths.save_by_completed_t);
        assert_eq!(
            checkpoint_save_path(
                &paths,
                ComputeCursor {
                    next_t: 61,
                    next_s: 0
                }
            ),
            Some(PathBuf::from("checkpoint/t60.checkpoint"))
        );

        let explicit = checkpoint_paths_for(
            &args(&["--checkpoint", "runs/local/resolution.checkpoint"]),
            100,
        )
        .unwrap();
        let explicit_path = Some(PathBuf::from("runs/local/resolution.checkpoint"));
        assert_eq!(explicit.load, explicit_path);
        assert_eq!(explicit.save, explicit_path);
        assert!(!explicit.auto_discover_load);
        assert!(!explicit.save_by_completed_t);
        assert_eq!(
            checkpoint_save_path(
                &explicit,
                ComputeCursor {
                    next_t: 61,
                    next_s: 0
                }
            ),
            Some(PathBuf::from("runs/local/resolution.checkpoint"))
        );
    }

    #[test]
    fn default_checkpoint_names_encode_t() {
        assert_eq!(checkpoint_degree_from_name("t0.checkpoint"), Some(0));
        assert_eq!(checkpoint_degree_from_name("t140.checkpoint"), Some(140));
        assert_eq!(checkpoint_degree_from_name("resolution.checkpoint"), None);
        assert_eq!(checkpoint_degree_from_name("t140.csv"), None);
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
