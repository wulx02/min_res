use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::hash::Hasher;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering as AtomicOrdering},
};

use serde::{Deserialize, Serialize};

use crate::fast_hash::FastHasher;
use crate::milnor::{CoeffKey, Milnor};
use crate::resolution::{ComputeCursor, Generator, ModuleTerm, Resolution, ResolutionSnapshot};

pub const SHARDED_CHECKPOINT_FORMAT: &str = "NMR_SHARDED_V1";
pub const SHARDED_DELTA_FORMAT: &str = "NMR_DELTA_SHARDED_V1";
pub const MINIMAL_RESOLUTION_JSONL_FORMAT: &str = "ext-minimal-resolution-jsonl";
pub const MINIMAL_RESOLUTION_JSONL_VERSION: u32 = 1;
const FORMAT_VERSION: u32 = 1;
const GEN_META_FILE: &str = "gen_meta.pack";
const DIFF_TERMS_FILE: &str = "diff_terms.pack";
const MANIFEST_FILE: &str = "manifest.json";
const GEN_META_RECORD_BYTES: usize = 5 * std::mem::size_of::<u64>();
const DIFF_TERM_RECORD_BYTES: usize = 2 * std::mem::size_of::<u64>();
const CHECKPOINT_FILES: [&str; 3] = [MANIFEST_FILE, GEN_META_FILE, DIFF_TERMS_FILE];
static TEMP_CHECKPOINT_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardedManifest {
    pub format_name: String,
    pub format_version: u32,
    pub kind: String,
    pub source_old_checkpoint_hash: Option<String>,
    pub source_git_commit: Option<String>,
    pub input_snapshot_hash: Option<String>,
    pub internal_degree: Option<usize>,
    pub completed_internal_degree: usize,
    pub next_internal_degree: usize,
    pub total_generator_count: usize,
    pub total_differential_term_count: usize,
    pub generator_id_min: Option<usize>,
    pub generator_id_max: Option<usize>,
    pub max_homological_degree: usize,
    pub num_workers: Option<usize>,
    pub expected_worker_ids: Vec<usize>,
    pub pack_hashes: Vec<String>,
    pub total_new_generators: Option<usize>,
    pub per_q_new_generator_counts: BTreeMap<usize, usize>,
    pub rank_convention: String,
    pub q_blocks: Vec<QBlockManifest>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct QBlockManifest {
    pub q: usize,
    pub gen_meta_offset: u64,
    pub gen_meta_len: u64,
    pub gen_count: usize,
    pub min_internal_degree: Option<usize>,
    pub max_internal_degree: Option<usize>,
    pub diff_terms_offset: u64,
    pub diff_terms_len: u64,
    pub diff_term_count: usize,
    pub gen_meta_hash: String,
    pub diff_terms_hash: String,
}

#[derive(Clone, Debug)]
pub struct ShardedLoadResult {
    pub snapshot: ResolutionSnapshot,
    pub total_generator_count: usize,
}

#[derive(Clone, Debug)]
pub struct ShardedVerifyReport {
    pub format_name: String,
    pub completed_internal_degree: usize,
    pub total_generator_count: usize,
    pub total_differential_term_count: usize,
    pub q_block_count: usize,
    pub max_homological_degree: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolutionJsonlExportSummary {
    pub generator_count: usize,
    pub differential_term_count: usize,
}

#[derive(Serialize)]
struct ResolutionJsonlMetadata<'a> {
    record_type: &'static str,
    format: &'static str,
    format_version: u32,
    prime: u8,
    coefficient_field: &'static str,
    algebra: &'static str,
    basis: &'static str,
    max_internal_degree: usize,
    generator_count: usize,
    differential_term_count: usize,
    rank_convention: &'static str,
    source_checkpoint: ResolutionJsonlCheckpoint<'a>,
}

#[derive(Serialize)]
struct ResolutionJsonlCheckpoint<'a> {
    format: &'a str,
    format_version: u32,
    completed_internal_degree: usize,
    source_git_commit: Option<&'a str>,
}

#[derive(Serialize)]
struct ResolutionJsonlGenerator {
    record_type: &'static str,
    format_version: u32,
    id: usize,
    name: String,
    s: usize,
    t: usize,
    stem: usize,
    differential: Vec<ResolutionJsonlDifferentialTerm>,
}

#[derive(Serialize)]
struct ResolutionJsonlDifferentialTerm {
    target_id: usize,
    target_name: String,
    milnor: Vec<u32>,
}

#[derive(Clone, Debug)]
struct GenMetaRecord {
    id: usize,
    q: usize,
    t: usize,
    diff_terms_offset: u64,
    term_count: usize,
}

#[derive(Clone, Debug)]
struct QBlockData {
    generators: Vec<Generator>,
}

pub fn manifest_path(dir: &Path) -> PathBuf {
    dir.join(MANIFEST_FILE)
}

pub fn read_manifest(dir: &Path) -> Result<ShardedManifest, String> {
    let path = manifest_path(dir);
    let file = File::open(&path)
        .map_err(|e| format!("failed to open sharded manifest {}: {e}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|e| format!("failed to parse sharded manifest {}: {e}", path.display()))
}

pub fn sharded_tree_size(dir: &Path) -> Result<u64, String> {
    let mut total = 0_u64;
    collect_tree_size(dir, &mut total)?;
    Ok(total)
}

pub fn write_sharded_checkpoint(
    out_dir: &Path,
    resolution: &Resolution,
    cursor: ComputeCursor,
    source_old_checkpoint_hash: Option<String>,
    source_git_commit: Option<String>,
    overwrite: bool,
) -> Result<ShardedManifest, String> {
    if out_dir.exists() && !overwrite {
        return Err(format!(
            "refusing to overwrite existing sharded directory {}; pass --overwrite",
            out_dir.display()
        ));
    }

    let temp_dir = create_checkpoint_temp_dir(out_dir)?;
    let manifest = match write_sharded_generators(
        &temp_dir,
        SHARDED_CHECKPOINT_FORMAT,
        "checkpoint",
        resolution.generators(),
        cursor.next_t.saturating_sub(1),
        cursor.next_t,
        source_old_checkpoint_hash,
        source_git_commit,
        None,
        None,
        None,
        Vec::new(),
        Vec::new(),
    ) {
        Ok(manifest) => manifest,
        Err(error) => {
            cleanup_checkpoint_temp_dir(&temp_dir);
            return Err(error);
        }
    };
    if let Err(error) = verify_sharded_checkpoint(&temp_dir) {
        cleanup_checkpoint_temp_dir(&temp_dir);
        return Err(format!(
            "refusing to publish invalid temporary checkpoint {}: {error}",
            temp_dir.display()
        ));
    }

    if out_dir.exists() {
        if !overwrite {
            cleanup_checkpoint_temp_dir(&temp_dir);
            return Err(format!(
                "refusing to overwrite existing sharded directory {}; pass --overwrite",
                out_dir.display()
            ));
        }
        if let Err(error) = validate_replaceable_checkpoint_dir(out_dir) {
            cleanup_checkpoint_temp_dir(&temp_dir);
            return Err(error);
        }
        if let Err(error) = atomic_exchange_checkpoint_dirs(&temp_dir, out_dir) {
            return Err(format!(
                "failed to atomically replace verified checkpoint {} with {}: {error}; the old checkpoint is unchanged and the new checkpoint remains in the temporary directory",
                out_dir.display(),
                temp_dir.display()
            ));
        }
        fs::remove_dir_all(&temp_dir).map_err(|error| {
            format!(
                "checkpoint {} was replaced successfully, but the old checkpoint could not be removed from temporary path {}: {error}",
                out_dir.display(),
                temp_dir.display()
            )
        })?;
        return Ok(manifest);
    }

    fs::rename(&temp_dir, out_dir).map_err(|error| {
        format!(
            "failed to move verified temporary checkpoint {} to {}: {error}; the new checkpoint remains in the temporary directory",
            temp_dir.display(),
            out_dir.display()
        )
    })?;
    Ok(manifest)
}

pub fn remove_verified_checkpoint(dir: &Path) -> Result<(), String> {
    validate_replaceable_checkpoint_dir(dir)?;
    fs::remove_dir_all(dir).map_err(|error| {
        format!(
            "failed to remove verified checkpoint {}: {error}",
            dir.display()
        )
    })
}

pub fn export_sharded_generators_csv(
    checkpoint_dir: &Path,
    max_t: usize,
    output: &Path,
) -> Result<usize, String> {
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create sharded export directory {}: {e}",
                parent.display()
            )
        })?;
    }

    let manifest = read_manifest(checkpoint_dir)?;
    ensure_format(&manifest, SHARDED_CHECKPOINT_FORMAT, checkpoint_dir)?;
    let mut seen_q = BTreeSet::new();
    let mut decoded_generators = 0_usize;
    let mut counts = BTreeMap::<(usize, usize), usize>::new();
    for block in &manifest.q_blocks {
        if !seen_q.insert(block.q) {
            return Err(format!(
                "sharded manifest {} contains duplicate q-block q={}",
                manifest_path(checkpoint_dir).display(),
                block.q
            ));
        }
        validate_record_block_len(
            checkpoint_dir,
            block.q,
            GEN_META_FILE,
            block.gen_meta_offset,
            block.gen_meta_len,
            GEN_META_RECORD_BYTES,
            block.gen_count,
            "gen_meta",
        )?;
        let meta = read_range(
            &checkpoint_dir.join(GEN_META_FILE),
            block.gen_meta_offset,
            block.gen_meta_len,
        )?;
        if hash_bytes(&meta) != block.gen_meta_hash {
            return Err(format!(
                "gen_meta hash mismatch for q={} in {}",
                block.q,
                checkpoint_dir.display()
            ));
        }
        let records = parse_gen_meta_records(block.q, &meta)?;
        decoded_generators = decoded_generators
            .checked_add(records.len())
            .ok_or_else(|| "manifest generator count overflowed usize".to_string())?;
        for record in records.into_iter().filter(|record| record.t <= max_t) {
            *counts.entry((record.q, record.t)).or_insert(0) += 1;
        }
    }
    if decoded_generators != manifest.total_generator_count {
        return Err(format!(
            "manifest total_generator_count={} but gen_meta q-blocks decode {} in {}",
            manifest.total_generator_count,
            decoded_generators,
            manifest_path(checkpoint_dir).display()
        ));
    }
    let tmp_path = output.with_extension("tmp");
    let mut writer = BufWriter::new(File::create(&tmp_path).map_err(|e| {
        format!(
            "failed to create temporary sharded export {}: {e}",
            tmp_path.display()
        )
    })?);
    writer
        .write_all(b"s,t,rank\n")
        .map_err(|e| format!("failed to write sharded export header: {e}"))?;
    for ((s, t), rank) in &counts {
        writeln!(writer, "{s},{t},{rank}")
            .map_err(|e| format!("failed to write sharded export row: {e}"))?;
    }
    writer
        .flush()
        .map_err(|e| format!("failed to flush sharded export {}: {e}", tmp_path.display()))?;
    fs::rename(&tmp_path, output).map_err(|e| {
        format!(
            "failed to replace sharded export {} with {}: {e}",
            output.display(),
            tmp_path.display()
        )
    })?;
    Ok(counts.len())
}

pub fn export_minimal_resolution_jsonl(
    checkpoint_dir: &Path,
    max_t: usize,
    output: &Path,
) -> Result<ResolutionJsonlExportSummary, String> {
    let manifest = read_manifest(checkpoint_dir)?;
    ensure_format(&manifest, SHARDED_CHECKPOINT_FORMAT, checkpoint_dir)?;
    if max_t > manifest.completed_internal_degree {
        return Err(format!(
            "checkpoint {} is complete only through t={}, so it cannot export t={max_t}",
            checkpoint_dir.display(),
            manifest.completed_internal_degree,
        ));
    }

    let all_qs = (0..=manifest.max_homological_degree).collect::<BTreeSet<_>>();
    let loaded = load_sparse_snapshot(checkpoint_dir, &[], &all_qs, &all_qs)?;
    let mut generators = loaded.snapshot.generators;
    generators.sort_by_key(|generator| generator.id);

    let names = generators
        .iter()
        .map(|generator| (generator.id, exported_generator_name(generator)))
        .collect::<BTreeMap<_, _>>();
    let selected = generators
        .iter()
        .filter(|generator| generator.t <= max_t)
        .collect::<Vec<_>>();
    let differential_term_count = selected
        .iter()
        .map(|generator| generator.differential.len())
        .sum();
    let summary = ResolutionJsonlExportSummary {
        generator_count: selected.len(),
        differential_term_count,
    };

    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create minimal-resolution export directory {}: {error}",
                parent.display()
            )
        })?;
    }
    let file_name = output
        .file_name()
        .ok_or_else(|| format!("invalid JSONL output path {}", output.display()))?
        .to_string_lossy();
    let tmp_path = output.with_file_name(format!("{file_name}.tmp"));
    let mut writer = BufWriter::new(File::create(&tmp_path).map_err(|error| {
        format!(
            "failed to create temporary JSONL export {}: {error}",
            tmp_path.display()
        )
    })?);

    let metadata = ResolutionJsonlMetadata {
        record_type: "metadata",
        format: MINIMAL_RESOLUTION_JSONL_FORMAT,
        format_version: MINIMAL_RESOLUTION_JSONL_VERSION,
        prime: 2,
        coefficient_field: "F2",
        algebra: "mod-2 Steenrod algebra",
        basis: "Milnor",
        max_internal_degree: max_t,
        generator_count: summary.generator_count,
        differential_term_count: summary.differential_term_count,
        rank_convention: "rank(s,t) is the number of minimal generators in bidegree (s,t), equal to dim_F2 Ext_A^{s,t}(F2,F2)",
        source_checkpoint: ResolutionJsonlCheckpoint {
            format: &manifest.format_name,
            format_version: manifest.format_version,
            completed_internal_degree: manifest.completed_internal_degree,
            source_git_commit: manifest.source_git_commit.as_deref(),
        },
    };
    write_jsonl_record(&mut writer, &metadata)?;

    for generator in selected {
        let stem = generator.t.checked_sub(generator.s).ok_or_else(|| {
            format!(
                "cannot export generator {} with s={} greater than t={}",
                generator.id, generator.s, generator.t
            )
        })?;
        let differential = generator
            .differential
            .iter()
            .map(|term| {
                let target_name = names.get(&term.generator).cloned().ok_or_else(|| {
                    format!(
                        "generator {} differential references missing generator {}",
                        generator.id, term.generator
                    )
                })?;
                Ok(ResolutionJsonlDifferentialTerm {
                    target_id: term.generator,
                    target_name,
                    milnor: Milnor::from_packed(term.coeff_packed).entries().to_vec(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let record = ResolutionJsonlGenerator {
            record_type: "generator",
            format_version: MINIMAL_RESOLUTION_JSONL_VERSION,
            id: generator.id,
            name: names[&generator.id].clone(),
            s: generator.s,
            t: generator.t,
            stem,
            differential,
        };
        write_jsonl_record(&mut writer, &record)?;
    }

    writer.flush().map_err(|error| {
        format!(
            "failed to flush JSONL export {}: {error}",
            tmp_path.display()
        )
    })?;
    fs::rename(&tmp_path, output).map_err(|error| {
        format!(
            "failed to replace JSONL export {} with {}: {error}",
            output.display(),
            tmp_path.display()
        )
    })?;
    Ok(summary)
}

fn exported_generator_name(generator: &Generator) -> String {
    if generator.id == 0 {
        "g0".to_string()
    } else {
        format!("g{}_{}_{}", generator.s, generator.t, generator.id)
    }
}

fn write_jsonl_record(writer: &mut impl Write, record: &impl Serialize) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, record)
        .map_err(|error| format!("failed to encode minimal-resolution JSONL record: {error}"))?;
    writer
        .write_all(b"\n")
        .map_err(|error| format!("failed to write minimal-resolution JSONL record: {error}"))
}

// Manifest provenance is intentionally explicit at this serialization
// boundary so fields cannot be silently omitted by a partial options object.
#[allow(clippy::too_many_arguments)]
fn write_sharded_generators(
    out_dir: &Path,
    format_name: &str,
    kind: &str,
    generators: &[Generator],
    completed_internal_degree: usize,
    next_internal_degree: usize,
    source_old_checkpoint_hash: Option<String>,
    source_git_commit: Option<String>,
    input_snapshot_hash: Option<String>,
    internal_degree: Option<usize>,
    num_workers: Option<usize>,
    expected_worker_ids: Vec<usize>,
    pack_hashes: Vec<String>,
) -> Result<ShardedManifest, String> {
    let gen_meta_path = out_dir.join(GEN_META_FILE);
    let diff_terms_path = out_dir.join(DIFF_TERMS_FILE);
    let mut gen_meta_writer = BufWriter::new(File::create(&gen_meta_path).map_err(|e| {
        format!(
            "failed to create sharded generator metadata {}: {e}",
            gen_meta_path.display()
        )
    })?);
    let mut diff_terms_writer = BufWriter::new(File::create(&diff_terms_path).map_err(|e| {
        format!(
            "failed to create sharded differential terms {}: {e}",
            diff_terms_path.display()
        )
    })?);

    let mut by_q: BTreeMap<usize, Vec<&Generator>> = BTreeMap::new();
    for generator in generators {
        by_q.entry(generator.s).or_default().push(generator);
    }
    for block in by_q.values_mut() {
        block.sort_by_key(|generator| generator.id);
    }

    let mut q_blocks = Vec::new();
    let mut gen_meta_offset = 0_u64;
    let mut diff_terms_offset = 0_u64;
    let mut total_terms = 0_usize;
    let mut per_q_new_generator_counts = BTreeMap::new();
    for (q, block_generators) in by_q {
        let block_gen_meta_offset = gen_meta_offset;
        let block_diff_terms_offset = diff_terms_offset;
        let mut meta_block = Vec::with_capacity(block_generators.len() * GEN_META_RECORD_BYTES);
        let mut diff_block = Vec::new();
        let mut min_t = None::<usize>;
        let mut max_t = None::<usize>;
        let mut block_term_count = 0_usize;

        for generator in block_generators {
            min_t = Some(
                min_t
                    .map(|value| value.min(generator.t))
                    .unwrap_or(generator.t),
            );
            max_t = Some(
                max_t
                    .map(|value| value.max(generator.t))
                    .unwrap_or(generator.t),
            );
            write_u64_to_vec(&mut meta_block, generator.id as u64);
            write_u64_to_vec(&mut meta_block, generator.s as u64);
            write_u64_to_vec(&mut meta_block, generator.t as u64);
            write_u64_to_vec(&mut meta_block, diff_terms_offset);
            write_u64_to_vec(&mut meta_block, generator.differential.len() as u64);
            for term in generator.differential.iter() {
                write_u64_to_vec(&mut diff_block, term.coeff_packed);
                write_u64_to_vec(&mut diff_block, term.generator as u64);
            }
            let terms_bytes = generator
                .differential
                .len()
                .checked_mul(DIFF_TERM_RECORD_BYTES)
                .ok_or_else(|| "differential term block is too large".to_string())?;
            diff_terms_offset = diff_terms_offset
                .checked_add(terms_bytes as u64)
                .ok_or_else(|| "differential term offsets overflowed u64".to_string())?;
            block_term_count += generator.differential.len();
        }

        gen_meta_writer.write_all(&meta_block).map_err(|e| {
            format!(
                "failed to write sharded generator metadata {}: {e}",
                gen_meta_path.display()
            )
        })?;
        diff_terms_writer.write_all(&diff_block).map_err(|e| {
            format!(
                "failed to write sharded differential terms {}: {e}",
                diff_terms_path.display()
            )
        })?;
        gen_meta_offset = gen_meta_offset
            .checked_add(meta_block.len() as u64)
            .ok_or_else(|| "generator metadata offsets overflowed u64".to_string())?;
        total_terms += block_term_count;
        per_q_new_generator_counts.insert(q, meta_block.len() / GEN_META_RECORD_BYTES);
        q_blocks.push(QBlockManifest {
            q,
            gen_meta_offset: block_gen_meta_offset,
            gen_meta_len: meta_block.len() as u64,
            gen_count: meta_block.len() / GEN_META_RECORD_BYTES,
            min_internal_degree: min_t,
            max_internal_degree: max_t,
            diff_terms_offset: block_diff_terms_offset,
            diff_terms_len: diff_block.len() as u64,
            diff_term_count: block_term_count,
            gen_meta_hash: hash_bytes(&meta_block),
            diff_terms_hash: hash_bytes(&diff_block),
        });
    }

    gen_meta_writer.flush().map_err(|e| {
        format!(
            "failed to flush sharded generator metadata {}: {e}",
            gen_meta_path.display()
        )
    })?;
    diff_terms_writer.flush().map_err(|e| {
        format!(
            "failed to flush sharded differential terms {}: {e}",
            diff_terms_path.display()
        )
    })?;

    let generator_id_min = generators.iter().map(|generator| generator.id).min();
    let generator_id_max = generators.iter().map(|generator| generator.id).max();
    let max_homological_degree = generators
        .iter()
        .map(|generator| generator.s)
        .max()
        .unwrap_or(0);
    let manifest = ShardedManifest {
        format_name: format_name.to_string(),
        format_version: FORMAT_VERSION,
        kind: kind.to_string(),
        source_old_checkpoint_hash,
        source_git_commit,
        input_snapshot_hash,
        internal_degree,
        completed_internal_degree,
        next_internal_degree,
        total_generator_count: generators.len(),
        total_differential_term_count: total_terms,
        generator_id_min,
        generator_id_max,
        max_homological_degree,
        num_workers,
        expected_worker_ids,
        pack_hashes,
        total_new_generators: (kind == "delta").then_some(generators.len()),
        per_q_new_generator_counts,
        rank_convention: "task_s output generators are recorded in homological degree q=s+1"
            .to_string(),
        q_blocks,
    };

    let manifest_path = manifest_path(out_dir);
    let manifest_file = File::create(&manifest_path)
        .map_err(|e| format!("failed to create manifest {}: {e}", manifest_path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(manifest_file), &manifest)
        .map_err(|e| format!("failed to write manifest {}: {e}", manifest_path.display()))?;
    Ok(manifest)
}

pub fn verify_sharded_checkpoint(dir: &Path) -> Result<ShardedVerifyReport, String> {
    let manifest = read_manifest(dir)?;
    if manifest.format_name != SHARDED_CHECKPOINT_FORMAT
        && manifest.format_name != SHARDED_DELTA_FORMAT
    {
        return Err(format!(
            "unsupported sharded format `{}` in {}",
            manifest.format_name,
            manifest_path(dir).display()
        ));
    }
    if manifest.format_version != FORMAT_VERSION {
        return Err(format!(
            "unsupported sharded format version {}",
            manifest.format_version
        ));
    }
    validate_manifest_storage(dir, &manifest)?;

    let mut all_generators = Vec::new();
    let mut counted_terms = 0_usize;
    for block in &manifest.q_blocks {
        let data = read_q_block(dir, &manifest, block.q, true)?;
        counted_terms += data
            .generators
            .iter()
            .map(|generator| generator.differential.len())
            .sum::<usize>();
        all_generators.extend(data.generators);
    }
    if all_generators.len() != manifest.total_generator_count {
        return Err(format!(
            "manifest total_generator_count={} but decoded {} generators",
            manifest.total_generator_count,
            all_generators.len()
        ));
    }
    if counted_terms != manifest.total_differential_term_count {
        return Err(format!(
            "manifest total_differential_term_count={} but decoded {} terms",
            manifest.total_differential_term_count, counted_terms
        ));
    }
    if manifest.format_name == SHARDED_CHECKPOINT_FORMAT {
        validate_generator_references(&all_generators)?;
    }
    Ok(ShardedVerifyReport {
        format_name: manifest.format_name,
        completed_internal_degree: manifest.completed_internal_degree,
        total_generator_count: manifest.total_generator_count,
        total_differential_term_count: manifest.total_differential_term_count,
        q_block_count: manifest.q_blocks.len(),
        max_homological_degree: manifest.max_homological_degree,
    })
}

pub fn load_sparse_snapshot(
    checkpoint_dir: &Path,
    delta_dirs: &[PathBuf],
    meta_qs: &BTreeSet<usize>,
    diff_qs: &BTreeSet<usize>,
) -> Result<ShardedLoadResult, String> {
    let mut generators = Vec::new();
    let mut total_generator_count = 0_usize;

    let checkpoint_manifest = read_manifest(checkpoint_dir)?;
    ensure_format(
        &checkpoint_manifest,
        SHARDED_CHECKPOINT_FORMAT,
        checkpoint_dir,
    )?;
    validate_manifest_storage(checkpoint_dir, &checkpoint_manifest)?;
    total_generator_count = total_generator_count.max(checkpoint_manifest.total_generator_count);
    let data = read_required_blocks(checkpoint_dir, &checkpoint_manifest, meta_qs, diff_qs)?;
    generators.extend(data.generators);

    for delta_dir in delta_dirs {
        let delta_manifest = read_manifest(delta_dir)?;
        ensure_format(&delta_manifest, SHARDED_DELTA_FORMAT, delta_dir)?;
        validate_manifest_storage(delta_dir, &delta_manifest)?;
        total_generator_count =
            total_generator_count.saturating_add(delta_manifest.total_generator_count);
        let data = read_required_blocks(delta_dir, &delta_manifest, meta_qs, diff_qs)?;
        generators.extend(data.generators);
    }

    generators.sort_by_key(|generator| generator.id);
    generators.dedup_by_key(|generator| generator.id);
    Ok(ShardedLoadResult {
        snapshot: ResolutionSnapshot { generators },
        total_generator_count,
    })
}

fn read_required_blocks(
    dir: &Path,
    manifest: &ShardedManifest,
    meta_qs: &BTreeSet<usize>,
    diff_qs: &BTreeSet<usize>,
) -> Result<QBlockData, String> {
    let mut generators = Vec::new();
    for &q in meta_qs {
        let data = read_q_block(dir, manifest, q, diff_qs.contains(&q))?;
        generators.extend(data.generators);
    }
    Ok(QBlockData { generators })
}

fn read_q_block(
    dir: &Path,
    manifest: &ShardedManifest,
    q: usize,
    include_diff: bool,
) -> Result<QBlockData, String> {
    let Some(block) = manifest.q_blocks.iter().find(|block| block.q == q) else {
        return Ok(QBlockData {
            generators: Vec::new(),
        });
    };
    validate_q_block_storage(dir, block)?;
    let meta = read_range(
        &dir.join(GEN_META_FILE),
        block.gen_meta_offset,
        block.gen_meta_len,
    )?;
    if hash_bytes(&meta) != block.gen_meta_hash {
        return Err(format!(
            "gen_meta hash mismatch for q={q} in {}",
            dir.display()
        ));
    }
    let records = parse_gen_meta_records(q, &meta)?;
    if records.len() != block.gen_count {
        return Err(format!(
            "manifest q={q} gen_count={} but decoded {} records",
            block.gen_count,
            records.len()
        ));
    }
    let diff = if include_diff {
        let bytes = read_range(
            &dir.join(DIFF_TERMS_FILE),
            block.diff_terms_offset,
            block.diff_terms_len,
        )?;
        if hash_bytes(&bytes) != block.diff_terms_hash {
            return Err(format!(
                "diff_terms hash mismatch for q={q} in {}",
                dir.display()
            ));
        }
        bytes
    } else {
        Vec::new()
    };
    let generators = records
        .into_iter()
        .map(|record| {
            let differential = if include_diff {
                terms_for_record(block, &diff, &record)?
            } else {
                Vec::new()
            };
            Ok(Generator {
                id: record.id,
                s: record.q,
                t: record.t,
                differential: Arc::new(differential),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(QBlockData { generators })
}

fn validate_manifest_storage(dir: &Path, manifest: &ShardedManifest) -> Result<(), String> {
    let mut seen_q = BTreeSet::new();
    let mut decoded_generators = 0_usize;
    let mut decoded_terms = 0_usize;
    for block in &manifest.q_blocks {
        if !seen_q.insert(block.q) {
            return Err(format!(
                "sharded manifest {} contains duplicate q-block q={}",
                manifest_path(dir).display(),
                block.q
            ));
        }
        validate_q_block_storage(dir, block)?;
        decoded_generators = decoded_generators
            .checked_add(block.gen_count)
            .ok_or_else(|| "manifest generator count overflowed usize".to_string())?;
        decoded_terms = decoded_terms
            .checked_add(block.diff_term_count)
            .ok_or_else(|| "manifest differential term count overflowed usize".to_string())?;
    }
    if decoded_generators != manifest.total_generator_count {
        return Err(format!(
            "manifest total_generator_count={} but q-blocks sum to {} in {}",
            manifest.total_generator_count,
            decoded_generators,
            manifest_path(dir).display()
        ));
    }
    if decoded_terms != manifest.total_differential_term_count {
        return Err(format!(
            "manifest total_differential_term_count={} but q-blocks sum to {} in {}",
            manifest.total_differential_term_count,
            decoded_terms,
            manifest_path(dir).display()
        ));
    }
    Ok(())
}

fn validate_q_block_storage(dir: &Path, block: &QBlockManifest) -> Result<(), String> {
    validate_record_block_len(
        dir,
        block.q,
        GEN_META_FILE,
        block.gen_meta_offset,
        block.gen_meta_len,
        GEN_META_RECORD_BYTES,
        block.gen_count,
        "gen_meta",
    )?;
    validate_record_block_len(
        dir,
        block.q,
        DIFF_TERMS_FILE,
        block.diff_terms_offset,
        block.diff_terms_len,
        DIFF_TERM_RECORD_BYTES,
        block.diff_term_count,
        "diff_terms",
    )
}

// The modulo form keeps this format check compatible with the minimum Rust
// version implied by edition 2024.
#[allow(clippy::too_many_arguments, clippy::manual_is_multiple_of)]
fn validate_record_block_len(
    dir: &Path,
    q: usize,
    file_name: &str,
    offset: u64,
    len: u64,
    record_bytes: usize,
    expected_records: usize,
    label: &str,
) -> Result<(), String> {
    if len % record_bytes as u64 != 0 {
        return Err(format!(
            "{label} block q={q} in {} has {len} bytes, not a multiple of {record_bytes}",
            dir.display()
        ));
    }
    let records = usize::try_from(len / record_bytes as u64).map_err(|_| {
        format!(
            "{label} block q={q} in {} has too many records for this platform",
            dir.display()
        )
    })?;
    if records != expected_records {
        return Err(format!(
            "manifest {label} block q={q} in {} records={}, but byte length implies {}",
            dir.display(),
            expected_records,
            records
        ));
    }
    validate_file_range(&dir.join(file_name), offset, len)
}

#[allow(clippy::manual_is_multiple_of)]
fn parse_gen_meta_records(q: usize, bytes: &[u8]) -> Result<Vec<GenMetaRecord>, String> {
    if bytes.len() % GEN_META_RECORD_BYTES != 0 {
        return Err(format!(
            "gen_meta block q={q} has {} bytes, not a multiple of {GEN_META_RECORD_BYTES}",
            bytes.len()
        ));
    }
    let mut records = Vec::with_capacity(bytes.len() / GEN_META_RECORD_BYTES);
    let (chunks, remainder) = bytes.as_chunks::<GEN_META_RECORD_BYTES>();
    debug_assert!(remainder.is_empty());
    for chunk in chunks {
        let id = u64_at(chunk, 0)? as usize;
        let record_q = u64_at(chunk, 8)? as usize;
        if record_q != q {
            return Err(format!(
                "gen_meta block q={q} contains record for q={record_q}"
            ));
        }
        records.push(GenMetaRecord {
            id,
            q: record_q,
            t: u64_at(chunk, 16)? as usize,
            diff_terms_offset: u64_at(chunk, 24)?,
            term_count: u64_at(chunk, 32)? as usize,
        });
    }
    Ok(records)
}

fn terms_for_record(
    block: &QBlockManifest,
    diff_block: &[u8],
    record: &GenMetaRecord,
) -> Result<Vec<ModuleTerm>, String> {
    if record.diff_terms_offset < block.diff_terms_offset {
        return Err(format!(
            "generator {} q={} has diff offset before q-block",
            record.id, record.q
        ));
    }
    let relative_offset = (record.diff_terms_offset - block.diff_terms_offset) as usize;
    let byte_len = record
        .term_count
        .checked_mul(DIFF_TERM_RECORD_BYTES)
        .ok_or_else(|| "differential record is too large".to_string())?;
    let end = relative_offset
        .checked_add(byte_len)
        .ok_or_else(|| "differential record offset overflow".to_string())?;
    let Some(bytes) = diff_block.get(relative_offset..end) else {
        return Err(format!(
            "generator {} q={} differential exceeds q-block",
            record.id, record.q
        ));
    };
    let mut terms = Vec::with_capacity(record.term_count);
    let (chunks, remainder) = bytes.as_chunks::<DIFF_TERM_RECORD_BYTES>();
    debug_assert!(remainder.is_empty());
    for chunk in chunks {
        terms.push(ModuleTerm {
            coeff_packed: u64_at(chunk, 0)? as CoeffKey,
            generator: u64_at(chunk, 8)? as usize,
        });
    }
    Ok(terms)
}

fn validate_generator_references(generators: &[Generator]) -> Result<(), String> {
    let mut by_id = BTreeMap::new();
    for generator in generators {
        if by_id.insert(generator.id, generator).is_some() {
            return Err(format!("duplicate generator id {}", generator.id));
        }
    }
    for generator in generators {
        for term in generator.differential.iter() {
            let target = by_id.get(&term.generator).ok_or_else(|| {
                format!(
                    "generator {} differential references missing target {}",
                    generator.id, term.generator
                )
            })?;
            if target.s + 1 != generator.s {
                return Err(format!(
                    "generator {} in q={} has target {} in q={}, expected q={}",
                    generator.id,
                    generator.s,
                    target.id,
                    target.s,
                    generator.s.saturating_sub(1)
                ));
            }
            let coeff_degree = Milnor::from_packed(term.coeff_packed).degree();
            if coeff_degree == 0 {
                return Err(format!(
                    "generator {} differential has unit coefficient on target {}",
                    generator.id, target.id
                ));
            }
            if target.t + coeff_degree != generator.t {
                return Err(format!(
                    "generator {} degree mismatch: coeff_degree={} target_t={} source_t={}",
                    generator.id, coeff_degree, target.t, generator.t
                ));
            }
        }
    }
    Ok(())
}

fn ensure_format(manifest: &ShardedManifest, expected: &str, dir: &Path) -> Result<(), String> {
    if manifest.format_name != expected {
        return Err(format!(
            "sharded directory {} has format `{}`, expected `{expected}`",
            dir.display(),
            manifest.format_name
        ));
    }
    if manifest.format_version != FORMAT_VERSION {
        return Err(format!(
            "sharded directory {} has format_version={}, expected {FORMAT_VERSION}",
            dir.display(),
            manifest.format_version
        ));
    }
    Ok(())
}

fn collect_tree_size(path: &Path, total: &mut u64) -> Result<(), String> {
    let metadata = fs::metadata(path)
        .map_err(|e| format!("failed to stat sharded path {}: {e}", path.display()))?;
    if metadata.is_file() {
        *total = total
            .checked_add(metadata.len())
            .ok_or_else(|| "sharded tree size overflowed u64".to_string())?;
        return Ok(());
    }
    if !metadata.is_dir() {
        return Err(format!(
            "sharded path {} is neither a file nor a directory",
            path.display()
        ));
    }
    for entry in fs::read_dir(path)
        .map_err(|e| format!("failed to read sharded directory {}: {e}", path.display()))?
    {
        let entry = entry.map_err(|e| format!("failed to read sharded directory entry: {e}"))?;
        collect_tree_size(&entry.path(), total)?;
    }
    Ok(())
}

fn create_checkpoint_temp_dir(out_dir: &Path) -> Result<PathBuf, String> {
    let file_name = out_dir.file_name().ok_or_else(|| {
        format!(
            "checkpoint output {} must name a directory, not a filesystem root or parent directory",
            out_dir.display()
        )
    })?;
    let parent = out_dir
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "failed to create checkpoint parent directory {}: {error}",
            parent.display()
        )
    })?;

    for _ in 0..100 {
        let counter = TEMP_CHECKPOINT_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let temp_name = format!(
            ".{}.tmp-{}-{counter}",
            file_name.to_string_lossy(),
            std::process::id()
        );
        let temp_dir = parent.join(temp_name);
        match fs::create_dir(&temp_dir) {
            Ok(()) => return Ok(temp_dir),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "failed to create temporary checkpoint directory {}: {error}",
                    temp_dir.display()
                ));
            }
        }
    }

    Err(format!(
        "failed to create a unique temporary checkpoint next to {} after 100 attempts",
        out_dir.display()
    ))
}

fn cleanup_checkpoint_temp_dir(temp_dir: &Path) {
    let _ = fs::remove_dir_all(temp_dir);
}

#[cfg(target_os = "macos")]
fn atomic_exchange_checkpoint_dirs(left: &Path, right: &Path) -> Result<(), std::io::Error> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let left = CString::new(left.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let right = CString::new(right.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let result = unsafe { libc::renamex_np(left.as_ptr(), right.as_ptr(), libc::RENAME_SWAP) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(target_os = "linux")]
fn atomic_exchange_checkpoint_dirs(left: &Path, right: &Path) -> Result<(), std::io::Error> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let left = CString::new(left.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let right = CString::new(right.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            left.as_ptr(),
            libc::AT_FDCWD,
            right.as_ptr(),
            libc::RENAME_EXCHANGE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn atomic_exchange_checkpoint_dirs(_left: &Path, _right: &Path) -> Result<(), std::io::Error> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "atomic checkpoint replacement is supported only on macOS and Linux",
    ))
}

fn validate_replaceable_checkpoint_dir(dir: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(dir).map_err(|error| {
        format!(
            "refusing to replace checkpoint directory {} because it cannot be inspected: {error}",
            dir.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "refusing to replace {} because it is not a regular checkpoint directory",
            dir.display()
        ));
    }

    let expected = CHECKPOINT_FILES.into_iter().collect::<BTreeSet<_>>();
    let mut found = BTreeSet::new();
    for entry in fs::read_dir(dir)
        .map_err(|error| format!("failed to inspect checkpoint {}: {error}", dir.display()))?
    {
        let entry =
            entry.map_err(|error| format!("failed to inspect checkpoint entry: {error}"))?;
        let name = entry.file_name().into_string().map_err(|_| {
            format!(
                "refusing to replace checkpoint {} because it contains a non-UTF-8 entry",
                dir.display()
            )
        })?;
        if !expected.contains(name.as_str()) {
            return Err(format!(
                "refusing to replace checkpoint {} because it contains unexpected entry {name:?}",
                dir.display()
            ));
        }
        let entry_metadata = fs::symlink_metadata(entry.path()).map_err(|error| {
            format!(
                "failed to inspect checkpoint entry {}: {error}",
                entry.path().display()
            )
        })?;
        if entry_metadata.file_type().is_symlink() || !entry_metadata.is_file() {
            return Err(format!(
                "refusing to replace checkpoint {} because entry {name:?} is not a regular file",
                dir.display()
            ));
        }
        found.insert(name);
    }
    for required in expected {
        if !found.contains(required) {
            return Err(format!(
                "refusing to replace checkpoint {} because required file {required:?} is missing",
                dir.display()
            ));
        }
    }

    let manifest = read_manifest(dir).map_err(|error| {
        format!(
            "refusing to replace directory {} because it is not a valid {SHARDED_CHECKPOINT_FORMAT} checkpoint: {error}",
            dir.display()
        )
    })?;
    ensure_format(&manifest, SHARDED_CHECKPOINT_FORMAT, dir).map_err(|error| {
        format!(
            "refusing to replace directory {} because it is not a valid {SHARDED_CHECKPOINT_FORMAT} checkpoint: {error}",
            dir.display()
        )
    })?;
    if manifest.kind != "checkpoint" {
        return Err(format!(
            "refusing to replace directory {} because manifest kind {:?} is not \"checkpoint\"",
            dir.display(),
            manifest.kind
        ));
    }
    verify_sharded_checkpoint(dir).map_err(|error| {
        format!(
            "refusing to replace directory {} because checkpoint verification failed: {error}",
            dir.display()
        )
    })?;
    Ok(())
}

fn read_range(path: &Path, offset: u64, len: u64) -> Result<Vec<u8>, String> {
    validate_file_range(path, offset, len)?;
    let mut file =
        File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("failed to seek {} to {offset}: {e}", path.display()))?;
    let len = usize::try_from(len)
        .map_err(|_| format!("range length in {} does not fit usize", path.display()))?;
    let mut bytes = vec![0_u8; len];
    file.read_exact(&mut bytes)
        .map_err(|e| format!("failed to read {len} bytes from {}: {e}", path.display()))?;
    Ok(bytes)
}

fn validate_file_range(path: &Path, offset: u64, len: u64) -> Result<(), String> {
    let file_len = fs::metadata(path)
        .map_err(|e| format!("failed to stat {} before reading: {e}", path.display()))?
        .len();
    let end = offset
        .checked_add(len)
        .ok_or_else(|| format!("range {offset}+{len} overflows u64 in {}", path.display()))?;
    if end > file_len {
        return Err(format!(
            "range offset={offset} len={len} exceeds file length {file_len} in {}",
            path.display()
        ));
    }
    Ok(())
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = FastHasher::default();
    hasher.write(bytes);
    format!("{:016x}", hasher.finish())
}

fn write_u64_to_vec(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let Some(slice) = bytes.get(offset..offset + 8) else {
        return Err("unexpected short u64 record".to_string());
    };
    let mut value = [0_u8; 8];
    value.copy_from_slice(slice);
    Ok(u64::from_le_bytes(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_test_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "ext_{label}_{}_{}",
            std::process::id(),
            TEMP_CHECKPOINT_COUNTER.fetch_add(1, AtomicOrdering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn verified_checkpoint_removal_refuses_unrelated_directory() {
        let root = fresh_test_root("checkpoint_removal_guard");
        let target = root.join("important");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("notes.txt"), b"keep me").unwrap();

        let error = remove_verified_checkpoint(&target).unwrap_err();
        assert!(error.contains("unexpected entry"));
        assert_eq!(fs::read(target.join("notes.txt")).unwrap(), b"keep me");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn verified_checkpoint_removal_removes_only_valid_checkpoint() {
        let root = fresh_test_root("checkpoint_removal_valid");
        let target = root.join("generated.checkpoint");
        write_sharded_checkpoint(
            &target,
            &Resolution::new(0),
            ComputeCursor::start(),
            None,
            None,
            false,
        )
        .unwrap();

        remove_verified_checkpoint(&target).unwrap();
        assert!(!target.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn checkpoint_replacement_refuses_unrelated_directory_without_deleting_it() {
        let root = fresh_test_root("checkpoint_unrelated_guard");
        let target = root.join("important");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("notes.txt"), b"keep me").unwrap();

        let error = write_sharded_checkpoint(
            &target,
            &Resolution::new(0),
            ComputeCursor::start(),
            None,
            None,
            true,
        )
        .unwrap_err();

        assert!(error.contains("unexpected entry"), "{error}");
        assert_eq!(fs::read(target.join("notes.txt")).unwrap(), b"keep me");
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn checkpoint_replacement_refuses_checkpoint_with_extra_files() {
        let root = fresh_test_root("checkpoint_extra_file_guard");
        let target = root.join("checkpoint.sharded");
        write_sharded_checkpoint(
            &target,
            &Resolution::new(0),
            ComputeCursor::start(),
            None,
            None,
            false,
        )
        .unwrap();
        fs::write(target.join("notes.txt"), b"keep me too").unwrap();

        let error = write_sharded_checkpoint(
            &target,
            &Resolution::new(3),
            ComputeCursor {
                next_t: 4,
                next_s: 0,
            },
            None,
            None,
            true,
        )
        .unwrap_err();

        assert!(error.contains("unexpected entry"), "{error}");
        assert_eq!(fs::read(target.join("notes.txt")).unwrap(), b"keep me too");
        assert_eq!(read_manifest(&target).unwrap().completed_internal_degree, 0);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_replacement_refuses_directory_symlink() {
        use std::os::unix::fs::symlink;

        let root = fresh_test_root("checkpoint_symlink_guard");
        let real_target = root.join("real-target");
        let link_target = root.join("linked-target");
        fs::create_dir(&real_target).unwrap();
        fs::write(real_target.join("notes.txt"), b"still here").unwrap();
        symlink(&real_target, &link_target).unwrap();

        let error = write_sharded_checkpoint(
            &link_target,
            &Resolution::new(0),
            ComputeCursor::start(),
            None,
            None,
            true,
        )
        .unwrap_err();

        assert!(
            error.contains("not a regular checkpoint directory"),
            "{error}"
        );
        assert_eq!(
            fs::read(real_target.join("notes.txt")).unwrap(),
            b"still here"
        );
        assert!(
            fs::symlink_metadata(&link_target)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn checkpoint_replacement_publishes_verified_tree_without_backup() {
        let root = fresh_test_root("checkpoint_safe_replace");
        let target = root.join("checkpoint.sharded");
        write_sharded_checkpoint(
            &target,
            &Resolution::new(0),
            ComputeCursor::start(),
            None,
            None,
            false,
        )
        .unwrap();

        write_sharded_checkpoint(
            &target,
            &Resolution::new(3),
            ComputeCursor {
                next_t: 4,
                next_s: 0,
            },
            None,
            None,
            true,
        )
        .unwrap();

        let manifest = read_manifest(&target).unwrap();
        assert_eq!(manifest.completed_internal_degree, 3);
        verify_sharded_checkpoint(&target).unwrap();
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        assert_eq!(
            fs::read_dir(&target).unwrap().count(),
            CHECKPOINT_FILES.len()
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn export_generators_csv_only_needs_manifest_and_gen_meta() {
        let root = std::env::temp_dir().join(format!(
            "ext_sharded_gen_meta_export_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let checkpoint = root.join("checkpoint.sharded");
        let csv = root.join("basis_t3.csv");
        let resolution = Resolution::new(3);
        write_sharded_checkpoint(
            &checkpoint,
            &resolution,
            ComputeCursor {
                next_t: 4,
                next_s: 0,
            },
            None,
            None,
            false,
        )
        .unwrap();
        fs::remove_file(checkpoint.join(DIFF_TERMS_FILE)).unwrap();

        let rows = export_sharded_generators_csv(&checkpoint, 3, &csv).unwrap();
        let text = fs::read_to_string(&csv).unwrap();
        assert_eq!(rows, 1);
        assert!(text.contains("s,t,rank\n"));
        assert!(text.contains("0,0,1\n"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn minimal_resolution_jsonl_is_versioned_and_contains_differentials() {
        let root = fresh_test_root("minimal_resolution_jsonl");
        let checkpoint = root.join("t4.checkpoint");
        let jsonl = root.join("t4.jsonl");
        let mut resolution = Resolution::new(4);
        resolution
            .compute(4, crate::resolution::ComputeMode::Naive)
            .unwrap();
        write_sharded_checkpoint(
            &checkpoint,
            &resolution,
            ComputeCursor {
                next_t: 5,
                next_s: 0,
            },
            None,
            Some("test-commit".to_string()),
            false,
        )
        .unwrap();

        let summary = export_minimal_resolution_jsonl(&checkpoint, 4, &jsonl).unwrap();
        let lines = fs::read_to_string(&jsonl)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), summary.generator_count + 1);

        let metadata = &lines[0];
        assert_eq!(metadata["record_type"], "metadata");
        assert_eq!(metadata["format"], MINIMAL_RESOLUTION_JSONL_FORMAT);
        assert_eq!(metadata["format_version"], MINIMAL_RESOLUTION_JSONL_VERSION);
        assert_eq!(metadata["generator_count"], summary.generator_count);
        assert_eq!(
            metadata["differential_term_count"],
            summary.differential_term_count
        );
        assert_eq!(
            metadata["source_checkpoint"]["source_git_commit"],
            "test-commit"
        );

        assert_eq!(lines[1]["record_type"], "generator");
        assert_eq!(lines[1]["name"], "g0");
        assert!(lines[1]["differential"].as_array().unwrap().is_empty());
        let nonzero_differential = lines[1..]
            .iter()
            .find(|record| !record["differential"].as_array().unwrap().is_empty())
            .expect("expected a nonzero differential in t <= 4");
        let first_term = &nonzero_differential["differential"][0];
        assert!(first_term["target_id"].is_u64());
        assert!(first_term["target_name"].is_string());
        assert!(first_term["milnor"].is_array());

        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/minimal-resolution-v1.schema.json"))
                .unwrap();
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn read_range_rejects_out_of_bounds_before_allocating() {
        let root = std::env::temp_dir().join(format!(
            "ext_sharded_read_range_guard_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let pack = root.join("tiny.pack");
        fs::write(&pack, [1_u8, 2, 3, 4]).unwrap();

        let err = read_range(&pack, 2, 3).unwrap_err();
        assert!(err.contains("exceeds file length"));

        let err = read_range(&pack, u64::MAX, 1).unwrap_err();
        assert!(err.contains("overflows u64"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn manifest_storage_rejects_inconsistent_q_block_lengths() {
        let root =
            std::env::temp_dir().join(format!("ext_sharded_manifest_guard_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join(GEN_META_FILE), vec![0_u8; GEN_META_RECORD_BYTES]).unwrap();
        fs::write(
            root.join(DIFF_TERMS_FILE),
            vec![0_u8; DIFF_TERM_RECORD_BYTES],
        )
        .unwrap();

        let bad_record_len = QBlockManifest {
            q: 0,
            gen_meta_offset: 0,
            gen_meta_len: GEN_META_RECORD_BYTES as u64 - 1,
            gen_count: 1,
            min_internal_degree: None,
            max_internal_degree: None,
            diff_terms_offset: 0,
            diff_terms_len: DIFF_TERM_RECORD_BYTES as u64,
            diff_term_count: 1,
            gen_meta_hash: String::new(),
            diff_terms_hash: String::new(),
        };
        let err = validate_q_block_storage(&root, &bad_record_len).unwrap_err();
        assert!(err.contains("not a multiple"));

        let bad_record_count = QBlockManifest {
            gen_meta_len: GEN_META_RECORD_BYTES as u64,
            gen_count: 2,
            ..bad_record_len
        };
        let err = validate_q_block_storage(&root, &bad_record_count).unwrap_err();
        assert!(err.contains("byte length implies"));

        let bad_range = QBlockManifest {
            gen_count: 1,
            gen_meta_offset: 1,
            gen_meta_len: GEN_META_RECORD_BYTES as u64,
            ..bad_record_count
        };
        let err = validate_q_block_storage(&root, &bad_range).unwrap_err();
        assert!(err.contains("exceeds file length"));

        let _ = fs::remove_dir_all(root);
    }
}
