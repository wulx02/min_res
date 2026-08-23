use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::{BTreeSet, BinaryHeap};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::f2::{
    BitVec, LinearSolver, homology_representative_batches_with_label, homology_representatives,
    homology_representatives_with_label,
};
use crate::fast_hash::{FastHashMap as HashMap, FastHashSet, fast_hash_map_with_capacity};
use crate::memory_probe::{ProcessMemory, log_process_memory};
use crate::milnor::{
    CoeffKey, Milnor, RowDecompOptions, basis_keys_of_degree, bounded_product_width,
    multiply_packed_fast_bounded_matching, multiply_packed_fast_raw_for_each_width,
    multiply_packed_keys_with_row_cache, packed_cmp, packed_support_width,
};
use crate::subalgebra::{Subalgebra, SubalgebraSelectionMode, subalgebra_selection_mode};

const DEFAULT_FULL_DIFFERENTIAL_CHUNK_BYTES: usize = 256 * 1024 * 1024;
const E0_PRECOMPUTE_CHUNK_PAIRS: usize = 1_000_000;
const E0_EMPTY_SEGMENT_COMPACT_LIMIT: usize = 64;
const E0_MISSING_SPARE_SHRINK_BYTES: usize = 64 * 1024 * 1024;
const PARALLEL_PRODUCT_SORT_MIN_KEYS: usize = 100_000;
const FIXED_T_BOUNDED_NAIVE_FALLBACK_MAX_MATRIX_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_FIXED_T_WORKER_PRODUCT_CACHE_MB: usize = 512;
const DEFAULT_FIXED_T_TASK_MEMORY_SAMPLE_SECS: u64 = 120;
pub const FIXED_T_BATCH_SKIP_EMPTY_DOMAIN_PATH: &str = "fixed-t-batch-skip-empty-domain";
pub const FIXED_T_BATCH_SKIP_ADAMS_VANISHING_PATH: &str = "fixed-t-batch-skip-adams-vanishing";
const STICKY_CONTIGUOUS_MAX_OVER_BASE_NUM: u128 = 11;
const STICKY_CONTIGUOUS_MAX_OVER_BASE_DEN: u128 = 10;

type ProductKey = (CoeffKey, CoeffKey);
type ProductKeyWithBound = (CoeffKey, CoeffKey, usize);
type E0ProfileKey = (u8, usize);
type Bidegree = (usize, usize);
type IndexedCoeff = (CoeffKey, usize);
type BasisIndex = HashMap<IndexedCoeff, usize>;
type BasisIndexCache = HashMap<Bidegree, Arc<BasisIndex>>;
type CoeffSignatureCacheKey = ((u8, usize, usize, usize, usize), usize, usize);
type CoeffSignatureBasisCache = HashMap<CoeffSignatureCacheKey, Arc<Vec<CoeffKey>>>;
type SignatureBasisIndexCache = HashMap<SignatureBasisKey, Arc<BasisIndex>>;

thread_local! {
    static FULL_DIFFERENTIAL_CHUNK_BYTES_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
    static FULL_DIFFERENTIAL_OUTPUT_CHUNK_BYTES_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
}

#[cfg(test)]
struct FullDifferentialChunkOverrideGuard {
    previous: Option<usize>,
}

#[cfg(test)]
impl Drop for FullDifferentialChunkOverrideGuard {
    fn drop(&mut self) {
        FULL_DIFFERENTIAL_CHUNK_BYTES_OVERRIDE.with(|override_bytes| {
            override_bytes.set(self.previous);
        });
    }
}

#[cfg(test)]
struct FullDifferentialOutputChunkOverrideGuard {
    previous: Option<usize>,
}

#[cfg(test)]
impl Drop for FullDifferentialOutputChunkOverrideGuard {
    fn drop(&mut self) {
        FULL_DIFFERENTIAL_OUTPUT_CHUNK_BYTES_OVERRIDE.with(|override_bytes| {
            override_bytes.set(self.previous);
        });
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Hash, Serialize)]
pub struct ModuleTerm {
    pub coeff_packed: CoeffKey,
    pub generator: usize,
}

#[derive(Clone, Debug)]
pub struct Generator {
    pub id: usize,
    pub s: usize,
    pub t: usize,
    pub differential: Arc<Vec<ModuleTerm>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct BasisElem {
    coeff_packed: CoeffKey,
    generator: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct SignatureBasisKey {
    s: usize,
    t: usize,
    subalgebra: (u8, usize, usize, usize, usize),
    sig_index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct SignatureRoutingKey {
    s: usize,
    t: usize,
    subalgebra: (u8, usize, usize, usize, usize),
}

#[derive(Clone, Debug)]
struct SignatureTranslation {
    to_zero: Vec<usize>,
    from_zero: Vec<usize>,
    current_len: usize,
    zero_len: usize,
    identity: bool,
}

#[derive(Clone)]
pub enum ComputeMode {
    Naive,
    FixedTBatch {
        workers: Option<usize>,
        shadow: bool,
        validate_commit: bool,
        scheduler: FixedTBatchScheduler,
        inner: Box<ComputeMode>,
    },
    Auto {
        subalgebras: Vec<Subalgebra>,
        strict: bool,
        force: bool,
    },
    AutoBoundedNaiveFallback {
        subalgebras: Vec<Subalgebra>,
        force: bool,
    },
    Accelerated {
        subalgebra: Subalgebra,
        strict: bool,
        force: bool,
    },
}

impl ComputeMode {
    fn ensure_subalgebra_signatures_through(
        &mut self,
        degree: usize,
        algebra_basis: &[Vec<CoeffKey>],
    ) -> Result<(), String> {
        match self {
            Self::Naive => Ok(()),
            Self::FixedTBatch { inner, .. } => {
                inner.ensure_subalgebra_signatures_through(degree, algebra_basis)
            }
            Self::Auto { subalgebras, .. } | Self::AutoBoundedNaiveFallback { subalgebras, .. } => {
                for subalgebra in subalgebras {
                    subalgebra.ensure_signatures_through(degree, algebra_basis)?;
                }
                Ok(())
            }
            Self::Accelerated { subalgebra, .. } => {
                subalgebra.ensure_signatures_through(degree, algebra_basis)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedTBatchScheduler {
    Contiguous,
    RoundRobin,
    WeightedContiguous,
    WeightedContiguousBuildCost,
    WeightedMinimaxBuildCost,
    LowPrefixBuildCost,
    AdaptiveContiguous,
    AdaptiveMinimaxContiguous,
    AdaptiveStickyContiguous,
    ProfileContiguous,
    WeightedGreedy,
    WeightedContiguousFrontMergeTailSplit,
}

impl FixedTBatchScheduler {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
            Self::RoundRobin => "round-robin",
            Self::WeightedContiguous => "weighted-contiguous",
            Self::WeightedContiguousBuildCost => "weighted-contiguous-build-cost",
            Self::WeightedMinimaxBuildCost => "minimax-build-cost",
            Self::LowPrefixBuildCost => "low-prefix-build-cost",
            Self::AdaptiveContiguous => "adaptive-contiguous",
            Self::AdaptiveMinimaxContiguous => "adaptive-minimax-contiguous",
            Self::AdaptiveStickyContiguous => "adaptive-sticky-contiguous",
            Self::ProfileContiguous => "profile-contiguous",
            Self::WeightedGreedy => "weighted-greedy",
            Self::WeightedContiguousFrontMergeTailSplit => {
                "weighted-contiguous-front-merge-tail-split"
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum E0CacheScope {
    Profile,
    Step,
}

impl E0CacheScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Profile => "profile",
            Self::Step => "step",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignatureMatrixCache {
    Keep,
    DropAfterUse,
}

impl SignatureMatrixCache {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::DropAfterUse => "drop",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum E0EmptyCachePolicy {
    SpeedHash,
    CompactSegments,
}

impl E0EmptyCachePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SpeedHash => "speed",
            Self::CompactSegments => "compact",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CachePolicy {
    pub e0_cache_scope: E0CacheScope,
    pub signature_matrix_cache: SignatureMatrixCache,
    pub e0_empty_cache: E0EmptyCachePolicy,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            e0_cache_scope: E0CacheScope::Profile,
            signature_matrix_cache: SignatureMatrixCache::DropAfterUse,
            e0_empty_cache: E0EmptyCachePolicy::SpeedHash,
        }
    }
}

static ALLOCATOR_RELIEF_ENABLED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
static SIGNATURE_LIFT_BATCH_TEST_HITS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static INITIAL_FULL_DIFFERENTIAL_BATCH_TEST_HITS: AtomicUsize = AtomicUsize::new(0);

pub fn set_allocator_relief_enabled(enabled: bool) {
    ALLOCATOR_RELIEF_ENABLED.store(enabled, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Default)]
struct SignatureSolveTiming {
    solver_lookup_s: f64,
    solve_s: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComputeCursor {
    pub next_t: usize,
    pub next_s: usize,
}

impl ComputeCursor {
    pub fn start() -> Self {
        Self {
            next_t: 1,
            next_s: 0,
        }
    }

    pub fn after_step(t: usize, s: usize) -> Self {
        match fixed_t_task_s_limit(t) {
            Some(limit) if s < limit => Self {
                next_t: t,
                next_s: s + 1,
            },
            _ => Self {
                next_t: t + 1,
                next_s: 0,
            },
        }
    }

    pub fn is_complete_for(self, max_t: usize) -> bool {
        self.next_t > max_t
    }
}

pub fn fixed_t_task_s_limit(t: usize) -> Option<usize> {
    t.checked_sub(1)
}

pub fn fixed_t_task_is_adams_vanishing(task_s: usize, t: usize) -> bool {
    let output_q = task_s.saturating_add(1);
    let Some(stem) = t.checked_sub(output_q) else {
        return false;
    };
    if stem == 0 {
        return false;
    }
    let epsilon = match output_q % 4 {
        0 | 1 => 1,
        2 => 2,
        3 => 3,
        _ => unreachable!(),
    };
    stem < output_q.saturating_mul(2).saturating_sub(epsilon)
}

#[derive(Clone, Debug)]
pub struct ResolutionSnapshot {
    pub generators: Vec<Generator>,
}

pub struct Resolution {
    algebra_basis: Arc<Vec<Vec<CoeffKey>>>,
    generators: Vec<Generator>,
    gens_by_s: Vec<Vec<usize>>,
    mult_packed_cache: HashMap<(CoeffKey, CoeffKey), Vec<CoeffKey>>,
    mult_e0_cache: HashMap<(CoeffKey, CoeffKey), ProductList>,
    mult_e0_empty_cache: EmptyProductCache,
    mult_e0_cache_profile: Option<E0ProfileKey>,
    mult_e0_profile_bank: HashMap<E0ProfileKey, E0ProfileProductCaches>,
    e0_profile_bank_limit_bytes: Option<usize>,
    row_decomp_cache: HashMap<(u32, usize), RowDecompOptions>,
    basis_cache: HashMap<(usize, usize), Arc<Vec<BasisElem>>>,
    basis_index_cache: BasisIndexCache,
    algebra_basis_index_cache: HashMap<usize, Arc<HashMap<CoeffKey, usize>>>,
    full_basis_lookup_cache: HashMap<(usize, usize), Arc<FullBasisLookup>>,
    coeff_signature_basis_cache: CoeffSignatureBasisCache,
    basis_signature_cache: HashMap<SignatureBasisKey, Arc<Vec<BasisElem>>>,
    basis_signature_index_cache: SignatureBasisIndexCache,
    basis_signature_routing_cache: HashMap<SignatureRoutingKey, Arc<Vec<(usize, usize)>>>,
    basis_signature_to_zero_cache: HashMap<SignatureBasisKey, Arc<SignatureTranslation>>,
    signature_matrix_cache: HashMap<SignatureBasisKey, Arc<LinearMap>>,
    signature_solver_cache: HashMap<SignatureBasisKey, Arc<LinearSolver>>,
    fixed_t_previous_task_micros: Option<(usize, HashMap<usize, u128>)>,
    fixed_t_previous_group_ranges: Option<(usize, Vec<(usize, usize)>)>,
    fixed_t_worker_product_caches: Vec<FixedTBatchPersistedProductCaches>,
    cache_policy: CachePolicy,
}

#[derive(Clone, Copy, Debug)]
pub struct ComputeProgress {
    /// Fixed-t task index. This is not the output homological degree; task `s`
    /// creates generators in homological degree `s + 1`.
    pub s: usize,
    pub t: usize,
    pub next_t: usize,
    pub next_s: usize,
    pub max_t: usize,
    pub generators: usize,
}

struct FrozenResolutionView {
    frozen_t: usize,
    algebra_basis: Arc<Vec<Vec<CoeffKey>>>,
    generators: Arc<Vec<Generator>>,
    gens_by_s: Arc<Vec<Vec<usize>>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FixedTBatchTaskResult {
    /// Fixed-t task index. The output homological degree is `output_q`.
    pub s: usize,
    /// Homological degree of generators created by this task.
    pub output_q: usize,
    pub t: usize,
    pub path: String,
    pub representatives: Vec<Vec<ModuleTerm>>,
    pub d_domain_dim: usize,
    pub d_target_dim: usize,
    pub boundary_domain_dim: usize,
    pub boundary_target_dim: usize,
    #[serde(default)]
    pub reduced_domain_dim: Option<usize>,
    #[serde(default)]
    pub reduced_target_dim: Option<usize>,
    #[serde(default)]
    pub reduced_boundary_domain_dim: Option<usize>,
    #[serde(default)]
    pub candidate_subalgebras: Vec<String>,
    #[serde(default)]
    pub chosen_subalgebra: Option<String>,
    pub matrix_s: f64,
    pub homology_s: f64,
    pub total_s: f64,
}

struct FixedTBatchGroupResult {
    group_index: usize,
    group_first: usize,
    group_last: usize,
    group_len: usize,
    wall_s: f64,
    task_sum_s: f64,
    task_max_s: f64,
    tasks: Vec<FixedTBatchTaskResult>,
    caches: FixedTBatchWorkerCaches,
    product_caches: FixedTBatchProductCaches,
}

#[derive(Default)]
struct FixedTBatchWorkerCaches {
    basis_cache: HashMap<(usize, usize), Arc<Vec<BasisElem>>>,
    basis_index_cache: BasisIndexCache,
    algebra_basis_index_cache: HashMap<usize, Arc<HashMap<CoeffKey, usize>>>,
    full_basis_lookup_cache: HashMap<(usize, usize), Arc<FullBasisLookup>>,
    coeff_signature_basis_cache: CoeffSignatureBasisCache,
    basis_signature_cache: HashMap<SignatureBasisKey, Arc<Vec<BasisElem>>>,
    basis_signature_index_cache: SignatureBasisIndexCache,
    basis_signature_routing_cache: HashMap<SignatureRoutingKey, Arc<Vec<(usize, usize)>>>,
    basis_signature_to_zero_cache: HashMap<SignatureBasisKey, Arc<SignatureTranslation>>,
    signature_matrix_cache: HashMap<SignatureBasisKey, Arc<LinearMap>>,
    signature_solver_cache: HashMap<SignatureBasisKey, Arc<LinearSolver>>,
}

#[derive(Default)]
struct FixedTBatchProductCaches {
    mult_packed_cache: HashMap<(CoeffKey, CoeffKey), Vec<CoeffKey>>,
    mult_e0_cache: HashMap<(CoeffKey, CoeffKey), ProductList>,
    mult_e0_empty_cache: EmptyProductCache,
    mult_e0_cache_profile: Option<E0ProfileKey>,
    mult_e0_profile_bank: HashMap<E0ProfileKey, E0ProfileProductCaches>,
    row_decomp_cache: HashMap<(u32, usize), RowDecompOptions>,
}

struct FixedTBatchPersistedProductCaches {
    group_first: usize,
    group_last: usize,
    caches: FixedTBatchProductCaches,
}

#[derive(Default)]
struct E0ProfileProductCaches {
    mult_e0_cache: HashMap<ProductKey, ProductList>,
    mult_e0_empty_cache: EmptyProductCache,
}

impl E0ProfileProductCaches {
    fn set_policy(&mut self, policy: E0EmptyCachePolicy) {
        self.mult_e0_empty_cache.set_policy(policy);
    }

    fn is_empty(&self) -> bool {
        self.mult_e0_cache.is_empty() && self.mult_e0_empty_cache.is_empty()
    }

    fn estimated_bytes(&self) -> usize {
        let (_, mult_e0_bytes) = estimate_product_list_map(&self.mult_e0_cache);
        mult_e0_bytes.saturating_add(self.mult_e0_empty_cache.estimated_bytes())
    }
}

impl FixedTBatchProductCaches {
    fn set_policy(&mut self, policy: E0EmptyCachePolicy) {
        self.mult_e0_empty_cache.set_policy(policy);
        for caches in self.mult_e0_profile_bank.values_mut() {
            caches.set_policy(policy);
        }
    }

    fn is_empty(&self) -> bool {
        self.mult_packed_cache.is_empty()
            && self.mult_e0_cache.is_empty()
            && self.mult_e0_empty_cache.is_empty()
            && self.mult_e0_profile_bank.is_empty()
            && self.row_decomp_cache.is_empty()
    }

    fn estimated_bytes(&self) -> usize {
        let (_, mult_packed_bytes) = estimate_vec_coeff_map(&self.mult_packed_cache);
        let (_, mult_e0_bytes) = estimate_product_list_map(&self.mult_e0_cache);
        let (_, _, row_decomp_bytes) = estimate_row_decomp_cache(&self.row_decomp_cache);
        mult_packed_bytes
            .saturating_add(mult_e0_bytes)
            .saturating_add(self.mult_e0_empty_cache.estimated_bytes())
            .saturating_add(estimate_e0_profile_bank_bytes(&self.mult_e0_profile_bank))
            .saturating_add(row_decomp_bytes)
    }

    fn e0_profile_bank_len(&self) -> usize {
        self.mult_e0_profile_bank.len()
    }

    fn e0_profile_bank_bytes(&self) -> usize {
        estimate_e0_profile_bank_bytes(&self.mult_e0_profile_bank)
    }

    fn shelve_active_profile(&mut self, policy: E0EmptyCachePolicy) {
        let Some(profile) = self.mult_e0_cache_profile.take() else {
            return;
        };
        let mut replacement_empty = EmptyProductCache::default();
        replacement_empty.set_policy(policy);
        let mut caches = E0ProfileProductCaches {
            mult_e0_cache: std::mem::take(&mut self.mult_e0_cache),
            mult_e0_empty_cache: std::mem::replace(
                &mut self.mult_e0_empty_cache,
                replacement_empty,
            ),
        };
        caches.set_policy(policy);
        if !caches.is_empty() {
            self.mult_e0_profile_bank.insert(profile, caches);
        }
    }

    fn trim_to_limit(&mut self, limit_bytes: usize, policy: E0EmptyCachePolicy) -> bool {
        self.shelve_active_profile(policy);
        if self.estimated_bytes() <= limit_bytes {
            return true;
        }

        let (_, mult_packed_bytes) = estimate_vec_coeff_map(&self.mult_packed_cache);
        let (_, _, row_decomp_bytes) = estimate_row_decomp_cache(&self.row_decomp_cache);
        let base_bytes = mult_packed_bytes.saturating_add(row_decomp_bytes);
        if base_bytes > limit_bytes {
            self.mult_e0_profile_bank.clear();
            return false;
        }

        let mut profiles = std::mem::take(&mut self.mult_e0_profile_bank)
            .into_iter()
            .map(|(profile, caches)| {
                let bytes = caches.estimated_bytes();
                (profile, bytes, caches)
            })
            .collect::<Vec<_>>();
        profiles.sort_by_key(|item| std::cmp::Reverse(item.1));

        let mut total_bytes = base_bytes;
        let mut kept_profiles = HashMap::default();
        for (profile, bytes, caches) in profiles {
            if total_bytes.saturating_add(bytes) <= limit_bytes {
                total_bytes = total_bytes.saturating_add(bytes);
                kept_profiles.insert(profile, caches);
            }
        }
        self.mult_e0_profile_bank = kept_profiles;
        self.mult_e0_cache.clear();
        self.mult_e0_cache.shrink_to_fit();
        self.mult_e0_empty_cache.clear_and_shrink();
        self.mult_e0_cache_profile = None;
        self.estimated_bytes() <= limit_bytes
    }
}

struct FrozenMatrixBuilder<'a> {
    view: &'a FrozenResolutionView,
    row_decomp_cache: HashMap<(u32, usize), RowDecompOptions>,
    mult_packed_cache: HashMap<(CoeffKey, CoeffKey), Vec<CoeffKey>>,
}

impl FrozenResolutionView {
    fn from_resolution(resolution: &Resolution, frozen_t: usize) -> Self {
        Self {
            frozen_t,
            algebra_basis: Arc::clone(&resolution.algebra_basis),
            generators: Arc::new(resolution.generators.clone()),
            gens_by_s: Arc::new(resolution.gens_by_s.clone()),
        }
    }

    fn generator(&self, id: usize) -> &Generator {
        &self.generators[id]
    }

    fn build_basis(&self, s: usize) -> Vec<BasisElem> {
        let mut out = Vec::new();
        let Some(gens) = self.gens_by_s.get(s) else {
            return out;
        };
        for &generator_id in gens {
            let generator = &self.generators[generator_id];
            if generator.t >= self.frozen_t {
                continue;
            }
            let alg_degree = self.frozen_t - generator.t;
            for &coeff_packed in &self.algebra_basis[alg_degree] {
                out.push(BasisElem {
                    coeff_packed,
                    generator: generator_id,
                });
            }
        }
        out
    }

    fn vector_to_terms(&self, s: usize, vector: &BitVec) -> Vec<ModuleTerm> {
        let basis = self.build_basis(s);
        vector
            .ones()
            .map(|i| ModuleTerm {
                coeff_packed: basis[i].coeff_packed,
                generator: basis[i].generator,
            })
            .collect()
    }

    fn compute_naive_homology_task(&self, s: usize) -> Result<FixedTBatchTaskResult, String> {
        let total_timer = Instant::now();
        let mut builder = FrozenMatrixBuilder::new(self);
        let matrix_timer = Instant::now();
        let d = builder.d_matrix(s)?;
        let boundaries = builder.d_matrix(s + 1)?;
        let matrix_s = matrix_timer.elapsed().as_secs_f64();

        let homology_timer = Instant::now();
        let representatives =
            homology_representatives(&d.columns, d.target_dim, &boundaries.columns, d.domain_dim)
                .into_iter()
                .map(|representative| self.vector_to_terms(s, &representative))
                .collect::<Vec<_>>();
        let homology_s = homology_timer.elapsed().as_secs_f64();

        Ok(FixedTBatchTaskResult {
            s,
            output_q: s + 1,
            t: self.frozen_t,
            path: "fixed-t-batch-naive-frozen".to_string(),
            representatives,
            d_domain_dim: d.domain_dim,
            d_target_dim: d.target_dim,
            boundary_domain_dim: boundaries.domain_dim,
            boundary_target_dim: boundaries.target_dim,
            reduced_domain_dim: None,
            reduced_target_dim: None,
            reduced_boundary_domain_dim: None,
            candidate_subalgebras: Vec::new(),
            chosen_subalgebra: None,
            matrix_s,
            homology_s,
            total_s: total_timer.elapsed().as_secs_f64(),
        })
    }
}

impl FrozenMatrixBuilder<'_> {
    fn new(view: &FrozenResolutionView) -> FrozenMatrixBuilder<'_> {
        FrozenMatrixBuilder {
            view,
            row_decomp_cache: HashMap::default(),
            mult_packed_cache: HashMap::default(),
        }
    }

    fn d_matrix(&mut self, s: usize) -> Result<LinearMap, String> {
        let domain = self.view.build_basis(s);
        let t = self.view.frozen_t;
        if s == 0 {
            let target_dim = usize::from(t == 0);
            let mut columns = Vec::with_capacity(domain.len());
            for elem in &domain {
                let mut col = BitVec::new(target_dim);
                if t == 0 && elem.generator == 0 && elem.coeff_packed == 0 {
                    col.set(0, true);
                }
                columns.push(col);
            }
            return Ok(LinearMap {
                columns,
                domain_dim: domain.len(),
                target_dim,
            });
        }

        let target = self.view.build_basis(s - 1);
        let target_index = basis_index(&target);
        let mut columns = Vec::with_capacity(domain.len());
        for elem in &domain {
            let col = self.differential_of_basis_elem_packed(elem, &target_index, target.len())?;
            columns.push(col);
        }
        Ok(LinearMap {
            columns,
            domain_dim: domain.len(),
            target_dim: target.len(),
        })
    }

    fn differential_of_basis_elem_packed(
        &mut self,
        elem: &BasisElem,
        target_index: &HashMap<(CoeffKey, usize), usize>,
        target_dim: usize,
    ) -> Result<BitVec, String> {
        let generator = &self.view.generators[elem.generator];
        let mut col = BitVec::new(target_dim);
        for term in generator.differential.iter() {
            let products = self
                .cached_multiply_packed_keys(elem.coeff_packed, term.coeff_packed)
                .to_vec();
            for coeff_packed in products {
                let image_key = (coeff_packed, term.generator);
                let Some(&row) = target_index.get(&image_key) else {
                    let coeff = Milnor::from_packed(coeff_packed);
                    return Err(format!(
                        "internal error: frozen image term {}*g{} missing from D_{},{}",
                        coeff, term.generator, self.view.frozen_t, self.view.frozen_t
                    ));
                };
                col.toggle(row);
            }
        }
        Ok(col)
    }

    fn cached_multiply_packed_keys(
        &mut self,
        left_key: CoeffKey,
        right_key: CoeffKey,
    ) -> &[CoeffKey] {
        let key = (left_key, right_key);
        if !self.mult_packed_cache.contains_key(&key) {
            let value = multiply_packed_keys_with_row_cache(
                left_key,
                right_key,
                &mut self.row_decomp_cache,
            );
            self.mult_packed_cache.insert(key, value);
        }
        self.mult_packed_cache
            .get(&key)
            .expect("frozen packed multiplication cache was just populated")
    }
}

impl Resolution {
    pub fn new(_max_t: usize) -> Self {
        let mut out = Self::empty();
        out.add_generator(0, 0, Vec::new());
        out
    }

    fn empty() -> Self {
        Self {
            algebra_basis: Arc::new(vec![basis_keys_of_degree(0)]),
            generators: Vec::new(),
            gens_by_s: Vec::new(),
            mult_packed_cache: HashMap::default(),
            mult_e0_cache: HashMap::default(),
            mult_e0_empty_cache: EmptyProductCache::default(),
            mult_e0_cache_profile: None,
            mult_e0_profile_bank: HashMap::default(),
            e0_profile_bank_limit_bytes: None,
            row_decomp_cache: HashMap::default(),
            basis_cache: HashMap::default(),
            basis_index_cache: HashMap::default(),
            algebra_basis_index_cache: HashMap::default(),
            full_basis_lookup_cache: HashMap::default(),
            coeff_signature_basis_cache: HashMap::default(),
            basis_signature_cache: HashMap::default(),
            basis_signature_index_cache: HashMap::default(),
            basis_signature_routing_cache: HashMap::default(),
            basis_signature_to_zero_cache: HashMap::default(),
            signature_matrix_cache: HashMap::default(),
            signature_solver_cache: HashMap::default(),
            fixed_t_previous_task_micros: None,
            fixed_t_previous_group_ranges: None,
            fixed_t_worker_product_caches: Vec::new(),
            cache_policy: CachePolicy::default(),
        }
    }

    fn ensure_algebra_basis_through(&mut self, degree: usize) {
        if self.algebra_basis.len() > degree {
            return;
        }
        let basis = Arc::make_mut(&mut self.algebra_basis);
        while basis.len() <= degree {
            basis.push(basis_keys_of_degree(basis.len()));
        }
    }

    pub fn set_cache_policy(&mut self, policy: CachePolicy) {
        self.cache_policy = policy;
        self.mult_e0_empty_cache.set_policy(policy.e0_empty_cache);
        for caches in self.mult_e0_profile_bank.values_mut() {
            caches.set_policy(policy.e0_empty_cache);
        }
    }

    #[cfg(test)]
    pub fn compute(&mut self, max_t: usize, mode: ComputeMode) -> Result<(), String> {
        self.compute_from_cursor(max_t, mode, ComputeCursor::start())
            .map(|_| ())
    }

    pub fn compute_from_cursor(
        &mut self,
        max_t: usize,
        mode: ComputeMode,
        cursor: ComputeCursor,
    ) -> Result<ComputeCursor, String> {
        self.compute_from_cursor_with_progress(max_t, mode, cursor, |_, _| Ok(()))
    }

    pub fn compute_from_cursor_with_progress(
        &mut self,
        max_t: usize,
        mut mode: ComputeMode,
        cursor: ComputeCursor,
        mut after_step: impl FnMut(&mut Self, ComputeProgress) -> Result<(), String>,
    ) -> Result<ComputeCursor, String> {
        if let ComputeMode::FixedTBatch {
            workers,
            shadow,
            validate_commit,
            scheduler,
            inner,
        } = mode
        {
            return self.compute_from_cursor_fixed_t_batch_with_progress(
                max_t,
                cursor,
                workers,
                shadow,
                validate_commit,
                scheduler,
                *inner,
                after_step,
            );
        }

        let mut t = cursor.next_t;
        let mut s = cursor.next_s;
        while t <= max_t {
            self.ensure_algebra_basis_through(t);
            mode.ensure_subalgebra_signatures_through(t, self.algebra_basis.as_ref())?;
            let Some(limit) = fixed_t_task_s_limit(t) else {
                t += 1;
                s = 0;
                continue;
            };
            if s > limit {
                t += 1;
                s = 0;
                continue;
            }

            self.compute_step(s, t, &mode)?;

            let next = ComputeCursor::after_step(t, s);
            after_step(
                self,
                ComputeProgress {
                    s,
                    t,
                    next_t: next.next_t,
                    next_s: next.next_s,
                    max_t,
                    generators: self.generators.len(),
                },
            )?;
            t = next.next_t;
            s = next.next_s;
        }
        Ok(ComputeCursor {
            next_t: max_t + 1,
            next_s: 0,
        })
    }

    fn compute_step(
        &mut self,
        s: usize,
        t: usize,
        mode: &ComputeMode,
    ) -> Result<(String, usize), String> {
        match mode {
            ComputeMode::Naive => Ok(("naive".to_string(), self.step_naive(s, t)?)),
            ComputeMode::FixedTBatch { .. } => {
                Err("fixed-t-batch is a layer algorithm, not a single bidegree step".to_string())
            }
            ComputeMode::Auto {
                subalgebras,
                strict,
                force,
            } => {
                if let Some(subalgebra) = self.choose_subalgebra(s, t, subalgebras, *force) {
                    let method = algorithm2_method_label(subalgebra, s, t);
                    Ok((method, self.step_algorithm2(s, t, subalgebra)?))
                } else if *strict {
                    let names = subalgebras
                        .iter()
                        .map(Subalgebra::name)
                        .collect::<Vec<_>>()
                        .join(",");
                    Err(format!(
                        "No selected subalgebra is justified at (s,t)=({s},{t}); candidates were {names}. Remove --strict to let Algorithm 1 fill this bidegree."
                    ))
                } else {
                    Ok(("naive fallback".to_string(), self.step_naive(s, t)?))
                }
            }
            ComputeMode::AutoBoundedNaiveFallback { subalgebras, force } => {
                if let Some(subalgebra) = self.choose_subalgebra(s, t, subalgebras, *force) {
                    let method = algorithm2_method_label(subalgebra, s, t);
                    Ok((method, self.step_algorithm2(s, t, subalgebra)?))
                } else {
                    let fallback_bytes = self.estimate_naive_homology_matrix_bytes(s, t);
                    if fallback_bytes <= FIXED_T_BOUNDED_NAIVE_FALLBACK_MAX_MATRIX_BYTES {
                        return Ok((
                            format!(
                                "bounded-naive fallback matrix_mb={:.1}",
                                bytes_to_mb(fallback_bytes)
                            ),
                            self.step_naive(s, t)?,
                        ));
                    }
                    let names = subalgebras
                        .iter()
                        .map(Subalgebra::name)
                        .collect::<Vec<_>>()
                        .join(",");
                    Err(format!(
                        "fixed-t batch refuses large full naive fallback at (s,t)=({s},{t}); no selected finite subalgebra is justified, estimated naive matrix memory is {:.1}MiB, and the fixed-t safety limit is {:.1}MiB. Candidates were {names}. Choose a different --subalgebras list, use --batch-inner accelerated --force for a controlled theorem-range experiment, or run low-dimensional diagnostics with --batch-inner naive.",
                        bytes_to_mb(fallback_bytes),
                        bytes_to_mb(FIXED_T_BOUNDED_NAIVE_FALLBACK_MAX_MATRIX_BYTES),
                    ))
                }
            }
            ComputeMode::Accelerated {
                subalgebra,
                strict,
                force,
            } => {
                if subalgebra_selection_is_disabled(subalgebra) {
                    return Err(format!(
                        "Requested subalgebra {} is disabled for selection and computation",
                        subalgebra.name()
                    ));
                }
                let valid = *force || subalgebra.selection_condition_applies(s, t);
                if valid {
                    Ok((
                        algorithm2_method_label(subalgebra, s, t),
                        self.step_algorithm2(s, t, subalgebra)?,
                    ))
                } else if *strict {
                    let bound = subalgebra.lower_line_bound(s);
                    Err(format!(
                        "Algorithm 2 with {} is not justified at (s,t)=({s},{t}); the Algorithm 2 lower-line condition needs t > {bound}. Remove --strict to bootstrap this bidegree with naive Algorithm 1, or choose another range/subalgebra.",
                        subalgebra.name()
                    ))
                } else {
                    Ok((
                        format!(
                            "bootstrap-naive; {} not in Algorithm 2 lower-line range",
                            subalgebra.name()
                        ),
                        self.step_naive(s, t)?,
                    ))
                }
            }
        }
    }

    pub fn snapshot(&self) -> ResolutionSnapshot {
        ResolutionSnapshot {
            generators: self.generators.clone(),
        }
    }

    // These options mirror the public fixed-t controls and are kept explicit
    // so the hot path does not allocate or clone a configuration object.
    #[allow(clippy::too_many_arguments)]
    fn compute_from_cursor_fixed_t_batch_with_progress(
        &mut self,
        max_t: usize,
        cursor: ComputeCursor,
        workers: Option<usize>,
        shadow: bool,
        validate_commit: bool,
        scheduler: FixedTBatchScheduler,
        mut inner: ComputeMode,
        mut after_step: impl FnMut(&mut Self, ComputeProgress) -> Result<(), String>,
    ) -> Result<ComputeCursor, String> {
        if cursor.next_s != 0 {
            return Err(format!(
                "fixed-t-batch checkpoints must resume at a t-layer boundary; got next_t={} next_s={}",
                cursor.next_t, cursor.next_s
            ));
        }
        if matches!(workers, Some(0)) {
            return Err("fixed-t-batch worker count must be positive".to_string());
        }

        let mut t = cursor.next_t;
        while t <= max_t {
            self.ensure_algebra_basis_through(t);
            inner.ensure_subalgebra_signatures_through(t, self.algebra_basis.as_ref())?;
            if shadow {
                self.compute_fixed_t_batch_shadow_layer(
                    t,
                    max_t,
                    workers,
                    validate_commit,
                    scheduler,
                    &inner,
                )?;
            } else {
                self.compute_fixed_t_batch_layer(t, workers, validate_commit, scheduler, &inner)?;
            }
            let next = ComputeCursor {
                next_t: t + 1,
                next_s: 0,
            };
            after_step(
                self,
                ComputeProgress {
                    s: fixed_t_task_s_limit(t).unwrap_or(0),
                    t,
                    next_t: next.next_t,
                    next_s: next.next_s,
                    max_t,
                    generators: self.generators.len(),
                },
            )?;
            t = next.next_t;
        }
        Ok(ComputeCursor {
            next_t: max_t + 1,
            next_s: 0,
        })
    }

    fn compute_fixed_t_batch_shadow_layer(
        &mut self,
        t: usize,
        max_t: usize,
        workers: Option<usize>,
        validate_commit: bool,
        scheduler: FixedTBatchScheduler,
        inner: &ComputeMode,
    ) -> Result<usize, String> {
        let start = self.snapshot();
        let mut sequential = Resolution::from_snapshot(max_t, start.clone())?;
        sequential.set_cache_policy(self.cache_policy);
        sequential.compute_from_cursor(
            t,
            inner.clone(),
            ComputeCursor {
                next_t: t,
                next_s: 0,
            },
        )?;

        let mut batch = Resolution::from_snapshot(max_t, start)?;
        batch.set_cache_policy(self.cache_policy);
        let added =
            batch.compute_fixed_t_batch_layer(t, workers, validate_commit, scheduler, inner)?;
        compare_fixed_t_shadow(t, &sequential, &batch)?;
        *self = batch;
        eprintln!("fixed_t_batch_shadow t={t} compare=ok added={added}");
        Ok(added)
    }

    fn compute_fixed_t_batch_layer(
        &mut self,
        t: usize,
        workers: Option<usize>,
        validate_commit: bool,
        scheduler: FixedTBatchScheduler,
        inner: &ComputeMode,
    ) -> Result<usize, String> {
        self.ensure_algebra_basis_through(t);
        let Some(s_limit) = fixed_t_task_s_limit(t) else {
            return Ok(0);
        };
        let basis_dims = (0..=s_limit + 1)
            .map(|s| self.basis_dimension_uncached(s, t))
            .collect::<Vec<_>>();
        let mut active_s = Vec::new();
        let mut skipped_empty_domain = Vec::new();
        for s in 0..=s_limit {
            let domain_dim = basis_dims[s];
            if domain_dim == 0 {
                skipped_empty_domain.push(empty_fixed_t_batch_task_result_from_dims(
                    s,
                    t,
                    &basis_dims,
                ));
            } else if fixed_t_task_is_adams_vanishing(s, t) {
                skipped_empty_domain.push(adams_vanishing_fixed_t_batch_task_result_from_dims(
                    s,
                    t,
                    &basis_dims,
                ));
            } else {
                active_s.push(s);
            }
        }
        let inner = inner.clone();
        let worker_count = workers.unwrap_or(1);
        let product_cache_limit_bytes = fixed_t_worker_product_cache_limit_bytes();
        let e0_profile_bank_enabled =
            product_cache_limit_bytes.is_some() && fixed_t_e0_profile_bank_enabled();
        let choice_dim_prewarm_enabled = worker_count > 1
            && fixed_t_choice_dim_prewarm_enabled()
            && fixed_t_auto_subalgebras(&inner).is_some();
        let frozen_view = if matches!(&inner, ComputeMode::Naive) || validate_commit {
            Some(FrozenResolutionView::from_resolution(self, t))
        } else {
            None
        };
        let detailed_logging = fixed_t_detailed_logging_enabled();
        if detailed_logging {
            eprintln!(
                "fixed_t_batch_layer_start t={t} active_s={} skipped_tasks={} skipped_empty_domain={} skipped_adams_vanishing={} workers={} scheduler={} worker_cache={} worker_product_cache_mb={} e0_profile_bank={} e0_scan_empty_skip={} choice_dim_prewarm={} validate_commit={} inner={} snapshot_generators={}",
                active_s.len(),
                skipped_empty_domain.len(),
                fixed_t_task_result_path_count(
                    &skipped_empty_domain,
                    FIXED_T_BATCH_SKIP_EMPTY_DOMAIN_PATH
                ),
                fixed_t_task_result_path_count(
                    &skipped_empty_domain,
                    FIXED_T_BATCH_SKIP_ADAMS_VANISHING_PATH
                ),
                worker_count,
                scheduler.as_str(),
                if worker_count == 1 {
                    "in-place"
                } else {
                    "shared-arc"
                },
                product_cache_limit_bytes
                    .map(|bytes| bytes_to_mb(bytes).to_string())
                    .unwrap_or_else(|| "off".to_string()),
                if e0_profile_bank_enabled { "on" } else { "off" },
                if e0_scan_empty_skip_enabled() {
                    "on"
                } else {
                    "off"
                },
                if choice_dim_prewarm_enabled {
                    "on"
                } else {
                    "off"
                },
                validate_commit,
                fixed_t_batch_inner_name(&inner),
                self.generator_count()
            );
            if let (Some(first), Some(last)) = fixed_t_task_result_path_bounds(
                &skipped_empty_domain,
                FIXED_T_BATCH_SKIP_EMPTY_DOMAIN_PATH,
            ) {
                eprintln!(
                    "fixed_t_batch_empty_domain_skipped t={t} count={} first_s={} last_s={}",
                    fixed_t_task_result_path_count(
                        &skipped_empty_domain,
                        FIXED_T_BATCH_SKIP_EMPTY_DOMAIN_PATH
                    ),
                    first.s,
                    last.s,
                );
            }
            if let (Some(first), Some(last)) = fixed_t_task_result_path_bounds(
                &skipped_empty_domain,
                FIXED_T_BATCH_SKIP_ADAMS_VANISHING_PATH,
            ) {
                eprintln!(
                    "fixed_t_batch_adams_vanishing_skipped t={t} count={} first_s={} last_s={} first_output_q={} last_output_q={}",
                    fixed_t_task_result_path_count(
                        &skipped_empty_domain,
                        FIXED_T_BATCH_SKIP_ADAMS_VANISHING_PATH
                    ),
                    first.s,
                    last.s,
                    first.output_q,
                    last.output_q,
                );
            }
        }
        log_process_memory("fixed_t_batch_layer_start", format!("t={t}"));
        if choice_dim_prewarm_enabled && !active_s.is_empty() {
            self.prewarm_fixed_t_choice_dims(t, &active_s, &inner);
        }

        let mut completed_group_ranges = Vec::<(usize, usize)>::new();
        let (mut results, mut worker_caches) = if matches!(&inner, ComputeMode::Naive) {
            let view = frozen_view
                .as_ref()
                .expect("frozen view is created for naive fixed-t batch");
            (
                active_s
                    .par_iter()
                    .map(|&s| view.compute_naive_homology_task(s))
                    .collect::<Result<Vec<_>, String>>()?,
                Vec::new(),
            )
        } else if active_s.is_empty() {
            (Vec::new(), Vec::new())
        } else if worker_count == 1 {
            if let (Some(&first), Some(&last)) = (active_s.first(), active_s.last()) {
                completed_group_ranges.push((first, last));
            }
            (
                self.compute_isolated_bidegree_group_in_place(&active_s, t, &inner, &basis_dims)?,
                Vec::new(),
            )
        } else {
            let groups = self.fixed_t_batch_groups(
                t,
                &active_s,
                worker_count,
                scheduler,
                &basis_dims,
                &inner,
            );
            completed_group_ranges = fixed_t_group_ranges(&groups);
            self.log_fixed_t_group_profile(t, scheduler, &active_s, &groups, &basis_dims, &inner);
            let product_caches = self.take_persistent_fixed_t_product_caches(
                &groups,
                product_cache_limit_bytes.is_some(),
            );
            // Each worker clone already shares immutable cache values through Arc;
            // avoid an extra seed clone that only duplicates cache-map tables.
            let seed: &Resolution = self;
            let group_results = groups
                .into_par_iter()
                .enumerate()
                .zip(product_caches.into_par_iter())
                .map(|((group_index, group), product_caches)| {
                    seed.compute_isolated_bidegree_group(
                        group_index,
                        &group,
                        t,
                        &inner,
                        &basis_dims,
                        product_caches,
                        product_cache_limit_bytes,
                        e0_profile_bank_enabled,
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            let mut tasks = Vec::new();
            let mut worker_caches = Vec::new();
            let mut next_product_caches = Vec::new();
            for group_result in group_results {
                log_fixed_t_group_done_profile(t, scheduler, &group_result);
                tasks.extend(group_result.tasks);
                worker_caches.push(group_result.caches);
                if product_cache_limit_bytes.is_some() {
                    next_product_caches.push(FixedTBatchPersistedProductCaches {
                        group_first: group_result.group_first,
                        group_last: group_result.group_last,
                        caches: group_result.product_caches,
                    });
                }
            }
            dedup_fixed_t_worker_cache_overlaps(&mut worker_caches, t);
            if product_cache_limit_bytes.is_some() {
                self.fixed_t_worker_product_caches = next_product_caches;
                self.log_persistent_fixed_t_product_cache_profile(t);
            } else {
                self.fixed_t_worker_product_caches.clear();
            }
            (tasks, worker_caches)
        };
        results.extend(skipped_empty_domain);
        results.sort_by_key(|result| result.s);
        if detailed_logging {
            for result in &results {
                if fixed_t_task_result_is_trivial_skip(result) {
                    continue;
                }
                eprintln!(
                    "fixed_t_batch_task t={} s={} path={} d_domain={} d_target={} boundary_domain={} boundary_target={} homology={} matrix_s={:.6} homology_s={:.6} total_s={:.6}",
                    result.t,
                    result.s,
                    result.path,
                    result.d_domain_dim,
                    result.d_target_dim,
                    result.boundary_domain_dim,
                    result.boundary_target_dim,
                    result.representatives.len(),
                    result.matrix_s,
                    result.homology_s,
                    result.total_s,
                );
            }
        }

        let commit_timer = Instant::now();
        let mut new_generator_ids = Vec::new();
        let mut added_total = 0;
        let mut invalidated_s = results
            .iter()
            .filter(|result| !result.representatives.is_empty())
            .map(|result| result.s + 1)
            .collect::<Vec<_>>();
        invalidated_s.sort_unstable();
        invalidated_s.dedup();
        for caches in &mut worker_caches {
            filter_fixed_t_worker_caches_for_commit(caches, t, &invalidated_s);
        }
        for result in &results {
            for differential in &result.representatives {
                if validate_commit {
                    let view = frozen_view
                        .as_ref()
                        .expect("frozen view is created when commit validation is enabled");
                    verify_minimal_terms_against_snapshot(t, differential, view)?;
                }
                let id = self.add_generator(result.s + 1, t, differential.to_vec());
                new_generator_ids.push(id);
                added_total += 1;
            }
        }
        if validate_commit {
            for &id in &new_generator_ids {
                self.verify_minimal_generator(id)?;
                self.verify_generator_d_squared(id)?;
            }
        }
        for caches in worker_caches {
            self.merge_fixed_t_worker_caches(caches);
        }
        self.update_fixed_t_task_time_history(t, &results);
        self.update_fixed_t_group_range_history(t, completed_group_ranges);
        if detailed_logging {
            eprintln!(
                "fixed_t_batch_commit t={t} added={added_total} validate_commit={} verify={} commit_s={:.6}",
                validate_commit,
                if validate_commit { "ok" } else { "skipped" },
                commit_timer.elapsed().as_secs_f64()
            );
        }
        log_process_memory(
            "fixed_t_batch_layer_end",
            format!("t={t} added={added_total}"),
        );
        Ok(added_total)
    }

    fn update_fixed_t_task_time_history(&mut self, t: usize, results: &[FixedTBatchTaskResult]) {
        let mut history = HashMap::<usize, u128>::default();
        for result in results {
            if fixed_t_task_result_is_trivial_skip(result) {
                continue;
            }
            let micros = if result.total_s.is_finite() && result.total_s > 0.0 {
                (result.total_s * 1_000_000.0).ceil() as u128
            } else {
                1
            };
            history.insert(result.s, micros.max(1));
        }
        self.fixed_t_previous_task_micros = Some((t, history));
    }

    fn update_fixed_t_group_range_history(&mut self, t: usize, mut ranges: Vec<(usize, usize)>) {
        ranges.retain(|(first, last)| first <= last);
        if ranges.is_empty() {
            self.fixed_t_previous_group_ranges = None;
            return;
        }
        ranges.sort_unstable();
        ranges.dedup();
        self.fixed_t_previous_group_ranges = Some((t, ranges));
    }

    fn fixed_t_batch_groups(
        &mut self,
        t: usize,
        active_s: &[usize],
        worker_count: usize,
        scheduler: FixedTBatchScheduler,
        basis_dims: &[usize],
        inner: &ComputeMode,
    ) -> Vec<Vec<usize>> {
        match scheduler {
            FixedTBatchScheduler::Contiguous => contiguous_groups(active_s, worker_count),
            FixedTBatchScheduler::RoundRobin => round_robin_groups(active_s, worker_count),
            FixedTBatchScheduler::WeightedContiguous => {
                let weights = self.fixed_t_scheduler_task_weights(t, active_s, basis_dims, inner);
                weighted_contiguous_groups(active_s, &weights, worker_count)
            }
            FixedTBatchScheduler::WeightedContiguousBuildCost => {
                let weights =
                    self.fixed_t_scheduler_build_cost_task_weights(t, active_s, basis_dims, inner);
                weighted_contiguous_groups(active_s, &weights, worker_count)
            }
            FixedTBatchScheduler::WeightedMinimaxBuildCost => {
                let weights =
                    self.fixed_t_scheduler_build_cost_task_weights(t, active_s, basis_dims, inner);
                minimax_contiguous_groups(active_s, &weights, worker_count)
            }
            FixedTBatchScheduler::LowPrefixBuildCost => {
                let weights =
                    self.fixed_t_scheduler_build_cost_task_weights(t, active_s, basis_dims, inner);
                low_prefix_build_cost_groups(active_s, &weights, worker_count)
            }
            FixedTBatchScheduler::AdaptiveContiguous => {
                let weights = self.adaptive_fixed_t_task_weights(t, active_s, basis_dims);
                weighted_contiguous_groups(active_s, &weights, worker_count)
            }
            FixedTBatchScheduler::AdaptiveMinimaxContiguous => {
                let weights = self.adaptive_fixed_t_task_weights(t, active_s, basis_dims);
                minimax_contiguous_groups(active_s, &weights, worker_count)
            }
            FixedTBatchScheduler::AdaptiveStickyContiguous => {
                let weights = self.adaptive_fixed_t_task_weights(t, active_s, basis_dims);
                let previous_ranges = self.adaptive_fixed_t_group_ranges(t);
                sticky_contiguous_groups(
                    active_s,
                    &weights,
                    worker_count,
                    previous_ranges.as_deref(),
                    STICKY_CONTIGUOUS_MAX_OVER_BASE_NUM,
                    STICKY_CONTIGUOUS_MAX_OVER_BASE_DEN,
                    t,
                )
            }
            FixedTBatchScheduler::ProfileContiguous => self
                .profile_contiguous_fixed_t_batch_groups(
                    t,
                    active_s,
                    worker_count,
                    basis_dims,
                    inner,
                ),
            FixedTBatchScheduler::WeightedGreedy => {
                let weights = self.fixed_t_scheduler_task_weights(t, active_s, basis_dims, inner);
                weighted_greedy_groups(active_s, &weights, worker_count)
            }
            FixedTBatchScheduler::WeightedContiguousFrontMergeTailSplit => {
                let weights = fixed_t_dimension_task_weights(active_s, basis_dims);
                weighted_contiguous_front_merge_tail_split_groups(active_s, &weights, worker_count)
            }
        }
    }

    fn fixed_t_scheduler_task_weights(
        &mut self,
        t: usize,
        active_s: &[usize],
        basis_dims: &[usize],
        inner: &ComputeMode,
    ) -> Vec<u128> {
        active_s
            .iter()
            .map(|&s| self.fixed_t_scheduler_task_weight(t, s, basis_dims, inner))
            .collect()
    }

    fn fixed_t_scheduler_build_cost_task_weights(
        &mut self,
        t: usize,
        active_s: &[usize],
        basis_dims: &[usize],
        inner: &ComputeMode,
    ) -> Vec<u128> {
        active_s
            .iter()
            .map(|&s| self.fixed_t_scheduler_build_cost_task_weight(t, s, basis_dims, inner))
            .collect()
    }

    fn fixed_t_scheduler_task_weight(
        &mut self,
        t: usize,
        s: usize,
        basis_dims: &[usize],
        inner: &ComputeMode,
    ) -> u128 {
        match inner {
            ComputeMode::Auto {
                subalgebras, force, ..
            }
            | ComputeMode::AutoBoundedNaiveFallback { subalgebras, force } => self
                .choose_subalgebra(s, t, subalgebras, *force)
                .map(|subalgebra| {
                    self.estimate_fixed_t_task_weight_from_signature_dims(s, t, subalgebra)
                })
                .unwrap_or_else(|| estimate_fixed_t_task_weight_from_dims(s, basis_dims)),
            ComputeMode::Accelerated {
                subalgebra, force, ..
            } if !subalgebra_selection_is_disabled(subalgebra)
                && (*force || subalgebra.selection_condition_applies(s, t)) =>
            {
                self.estimate_fixed_t_task_weight_from_signature_dims(s, t, subalgebra)
            }
            _ => estimate_fixed_t_task_weight_from_dims(s, basis_dims),
        }
    }

    fn fixed_t_scheduler_build_cost_task_weight(
        &mut self,
        t: usize,
        s: usize,
        basis_dims: &[usize],
        inner: &ComputeMode,
    ) -> u128 {
        match inner {
            ComputeMode::Auto {
                subalgebras, force, ..
            }
            | ComputeMode::AutoBoundedNaiveFallback { subalgebras, force } => self
                .choose_subalgebra(s, t, subalgebras, *force)
                .map(|subalgebra| {
                    self.estimate_fixed_t_task_build_cost_from_signature_pairs(s, t, subalgebra)
                })
                .unwrap_or_else(|| estimate_fixed_t_task_weight_from_dims(s, basis_dims)),
            ComputeMode::Accelerated {
                subalgebra, force, ..
            } if !subalgebra_selection_is_disabled(subalgebra)
                && (*force || subalgebra.selection_condition_applies(s, t)) =>
            {
                self.estimate_fixed_t_task_build_cost_from_signature_pairs(s, t, subalgebra)
            }
            _ => estimate_fixed_t_task_weight_from_dims(s, basis_dims),
        }
    }

    fn estimate_fixed_t_task_weight_from_signature_dims(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
    ) -> u128 {
        let domain = self.basis_signature_dim(s, t, subalgebra, 0);
        let target = if s == 0 {
            usize::from(t == 0)
        } else {
            self.basis_signature_dim(s - 1, t, subalgebra, 0)
        };
        let boundary_domain = self.basis_signature_dim(s + 1, t, subalgebra, 0);
        estimate_fixed_t_task_weight_from_dimensions(domain, target, boundary_domain)
    }

    fn estimate_fixed_t_task_build_cost_from_signature_pairs(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
    ) -> u128 {
        let domain_work = self.estimate_fixed_t_chain_group_build_cost(s, t, subalgebra);
        let boundary_work = self.estimate_fixed_t_chain_group_build_cost(s + 1, t, subalgebra);
        apply_fixed_t_subalgebra_build_cost_multiplier(
            domain_work.saturating_add(boundary_work),
            subalgebra,
        )
        .max(1)
    }

    fn estimate_fixed_t_chain_group_build_cost(
        &mut self,
        p: usize,
        t: usize,
        subalgebra: &Subalgebra,
    ) -> u128 {
        let Some(gens_len) = self.gens_by_s.get(p).map(Vec::len) else {
            return 0;
        };
        let mut total = 0_u128;
        for index in 0..gens_len {
            let generator_id = self.gens_by_s[p][index];
            let generator_t = self.generators[generator_id].t;
            if generator_t > t {
                continue;
            }
            let differential_len = self.generators[generator_id].differential.len() as u128;
            if differential_len == 0 {
                continue;
            }
            let q_b = self
                .coeff_signature_basis_cached(t - generator_t, subalgebra, 0)
                .len() as u128;
            total = total.saturating_add(q_b.saturating_mul(differential_len));
        }
        total
    }

    fn profile_contiguous_fixed_t_batch_groups(
        &mut self,
        t: usize,
        active_s: &[usize],
        worker_count: usize,
        basis_dims: &[usize],
        inner: &ComputeMode,
    ) -> Vec<Vec<usize>> {
        let weights = self.adaptive_fixed_t_task_weights(t, active_s, basis_dims);
        let Some((subalgebras, force)) = fixed_t_auto_subalgebras(inner) else {
            return weighted_contiguous_groups(active_s, &weights, worker_count);
        };
        let profiles = active_s
            .iter()
            .map(|&s| {
                self.choose_subalgebra(s, t, subalgebras, force)
                    .map(Subalgebra::cache_key)
            })
            .collect::<Vec<_>>();
        profile_contiguous_groups(active_s, &weights, &profiles, worker_count)
    }

    fn estimate_naive_homology_matrix_bytes(&self, s: usize, t: usize) -> usize {
        let domain = self.basis_dimension_uncached(s, t);
        let target = if s == 0 {
            usize::from(t == 0)
        } else {
            self.basis_dimension_uncached(s - 1, t)
        };
        let boundary_domain = self.basis_dimension_uncached(s + 1, t);
        domain
            .saturating_mul(BitVec::storage_bytes_for_len(target))
            .saturating_add(boundary_domain.saturating_mul(BitVec::storage_bytes_for_len(domain)))
    }

    fn basis_dimension_uncached(&self, s: usize, t: usize) -> usize {
        let Some(gens) = self.gens_by_s.get(s) else {
            return 0;
        };
        gens.iter()
            .filter_map(|&generator_id| {
                let generator = &self.generators[generator_id];
                if generator.t <= t {
                    Some(self.algebra_basis[t - generator.t].len())
                } else {
                    None
                }
            })
            .sum()
    }

    fn prewarm_fixed_t_choice_dims(&mut self, t: usize, active_s: &[usize], inner: &ComputeMode) {
        let Some((subalgebras, force)) = fixed_t_auto_subalgebras(inner) else {
            return;
        };
        let timer = Instant::now();
        let before_entries = self.coeff_signature_basis_cache.len();
        let before_bytes = estimate_arc_vec_cache(&self.coeff_signature_basis_cache);
        let mut dim_queries = 0usize;
        let mut candidate_tasks = 0usize;
        for &s in active_s {
            for subalgebra in subalgebras {
                if subalgebra_selection_is_disabled(subalgebra) {
                    continue;
                }
                if !(force || subalgebra.selection_condition_applies(s, t)) {
                    continue;
                }
                candidate_tasks += 1;
                let _ = self.basis_signature_dim(s, t, subalgebra, 0);
                dim_queries += 1;
                if s > 0 {
                    let _ = self.basis_signature_dim(s - 1, t, subalgebra, 0);
                    dim_queries += 1;
                }
                let _ = self.basis_signature_dim(s + 1, t, subalgebra, 0);
                dim_queries += 1;
            }
        }
        let after_entries = self.coeff_signature_basis_cache.len();
        let after_bytes = estimate_arc_vec_cache(&self.coeff_signature_basis_cache);
        if fixed_t_detailed_logging_enabled() {
            eprintln!(
                "fixed_t_choice_dim_prewarm t={t} active_s={} candidate_tasks={candidate_tasks} dim_queries={dim_queries} coeff_cache_added={} coeff_cache_mb_added={:.1} total_s={:.6}",
                active_s.len(),
                after_entries.saturating_sub(before_entries),
                bytes_to_mb(after_bytes.saturating_sub(before_bytes)),
                timer.elapsed().as_secs_f64(),
            );
        }
    }

    fn adaptive_fixed_t_task_weights(
        &self,
        t: usize,
        active_s: &[usize],
        basis_dims: &[usize],
    ) -> Vec<u128> {
        self.adaptive_fixed_t_task_weights_with_profile(t, active_s, basis_dims, true)
    }

    fn adaptive_fixed_t_task_weights_with_profile(
        &self,
        t: usize,
        active_s: &[usize],
        basis_dims: &[usize],
        log_profile: bool,
    ) -> Vec<u128> {
        let dim_weights = fixed_t_dimension_task_weights(active_s, basis_dims);
        let Some((history_t, history)) = &self.fixed_t_previous_task_micros else {
            return dim_weights;
        };
        if history_t.checked_add(1) != Some(t) || history.is_empty() {
            return dim_weights;
        }

        let mut common_time_sum = 0u128;
        let mut common_dim_sum = 0u128;
        for (&s, &dim_weight) in active_s.iter().zip(&dim_weights) {
            if let Some(&micros) = history.get(&s) {
                common_time_sum = common_time_sum.saturating_add(micros.max(1));
                common_dim_sum = common_dim_sum.saturating_add(dim_weight.max(1));
            }
        }
        let scaled_missing = |dim_weight: u128| -> u128 {
            if common_time_sum == 0 || common_dim_sum == 0 {
                dim_weight.max(1)
            } else {
                dim_weight
                    .saturating_mul(common_time_sum)
                    .div_ceil(common_dim_sum)
                    .max(1)
            }
        };
        let weights = active_s
            .iter()
            .zip(dim_weights.iter().copied())
            .map(|(&s, dim_weight)| {
                history
                    .get(&s)
                    .copied()
                    .map(|micros| micros.max(1))
                    .unwrap_or_else(|| scaled_missing(dim_weight))
            })
            .collect::<Vec<_>>();
        if log_profile && std::env::var_os("EXT_PROFILE_FIXED_T_GROUPS").is_some() {
            eprintln!(
                "fixed_t_adaptive_scheduler t={t} history_t={history_t} active_s={} history_entries={} common_entries={} common_time_us={} common_dim_weight={}",
                active_s.len(),
                history.len(),
                active_s
                    .iter()
                    .filter(|&&s| history.contains_key(&s))
                    .count(),
                common_time_sum,
                common_dim_sum,
            );
        }
        weights
    }

    fn adaptive_fixed_t_group_ranges(&self, t: usize) -> Option<Vec<(usize, usize)>> {
        let (history_t, ranges) = self.fixed_t_previous_group_ranges.as_ref()?;
        if history_t.checked_add(1) != Some(t) || ranges.is_empty() {
            return None;
        }
        Some(ranges.clone())
    }

    fn log_fixed_t_group_profile(
        &mut self,
        t: usize,
        scheduler: FixedTBatchScheduler,
        active_s: &[usize],
        groups: &[Vec<usize>],
        basis_dims: &[usize],
        inner: &ComputeMode,
    ) {
        if std::env::var_os("EXT_PROFILE_FIXED_T_GROUPS").is_none() {
            return;
        }
        let (weights, weight_source) =
            self.fixed_t_group_profile_weights(t, active_s, scheduler, basis_dims, inner);
        let weight_by_s = active_s
            .iter()
            .copied()
            .zip(weights)
            .collect::<HashMap<usize, u128>>();
        for (index, group) in groups.iter().enumerate() {
            let estimated_weight = group
                .iter()
                .map(|s| weight_by_s.get(s).copied().unwrap_or(1))
                .sum::<u128>();
            let dim_weight = group
                .iter()
                .map(|&s| estimate_fixed_t_task_weight_from_dims(s, basis_dims))
                .sum::<u128>();
            eprintln!(
                "fixed_t_batch_group t={t} scheduler={} group_index={index} group_first={} group_last={} group_len={} estimated_weight={estimated_weight} weight_source={weight_source} dim_weight={dim_weight}",
                scheduler.as_str(),
                group.first().copied().unwrap_or(0),
                group.last().copied().unwrap_or(0),
                group.len(),
            );
        }
    }

    fn fixed_t_group_profile_weights(
        &mut self,
        t: usize,
        active_s: &[usize],
        scheduler: FixedTBatchScheduler,
        basis_dims: &[usize],
        inner: &ComputeMode,
    ) -> (Vec<u128>, &'static str) {
        match scheduler {
            FixedTBatchScheduler::AdaptiveContiguous
            | FixedTBatchScheduler::AdaptiveMinimaxContiguous
            | FixedTBatchScheduler::AdaptiveStickyContiguous
            | FixedTBatchScheduler::ProfileContiguous => {
                let source = self.fixed_t_adaptive_weight_source(t, active_s);
                (
                    self.adaptive_fixed_t_task_weights_with_profile(t, active_s, basis_dims, false),
                    source,
                )
            }
            FixedTBatchScheduler::WeightedContiguous | FixedTBatchScheduler::WeightedGreedy => (
                self.fixed_t_scheduler_task_weights(t, active_s, basis_dims, inner),
                "signature-reduced",
            ),
            FixedTBatchScheduler::WeightedContiguousBuildCost => (
                self.fixed_t_scheduler_build_cost_task_weights(t, active_s, basis_dims, inner),
                "signature-build-cost",
            ),
            FixedTBatchScheduler::WeightedMinimaxBuildCost => (
                self.fixed_t_scheduler_build_cost_task_weights(t, active_s, basis_dims, inner),
                "signature-build-cost-minimax",
            ),
            FixedTBatchScheduler::LowPrefixBuildCost => (
                self.fixed_t_scheduler_build_cost_task_weights(t, active_s, basis_dims, inner),
                "signature-build-cost-low-prefix",
            ),
            FixedTBatchScheduler::Contiguous
            | FixedTBatchScheduler::RoundRobin
            | FixedTBatchScheduler::WeightedContiguousFrontMergeTailSplit => (
                fixed_t_dimension_task_weights(active_s, basis_dims),
                "dimension",
            ),
        }
    }

    fn fixed_t_adaptive_weight_source(&self, t: usize, active_s: &[usize]) -> &'static str {
        let Some((history_t, history)) = &self.fixed_t_previous_task_micros else {
            return "dimension";
        };
        if history_t.checked_add(1) != Some(t) || history.is_empty() {
            return "dimension";
        }
        if active_s.iter().any(|s| history.contains_key(s)) {
            "adaptive"
        } else {
            "dimension"
        }
    }

    fn clone_with_shared_worker_caches(&self) -> Self {
        self.clone_for_isolated_worker(true)
    }

    fn clone_for_isolated_worker(&self, inherit_worker_caches: bool) -> Self {
        let mut mult_e0_empty_cache = EmptyProductCache::default();
        mult_e0_empty_cache.set_policy(self.cache_policy.e0_empty_cache);
        Self {
            algebra_basis: Arc::clone(&self.algebra_basis),
            generators: self.generators.clone(),
            gens_by_s: self.gens_by_s.clone(),
            mult_packed_cache: HashMap::default(),
            mult_e0_cache: HashMap::default(),
            mult_e0_empty_cache,
            mult_e0_cache_profile: None,
            mult_e0_profile_bank: HashMap::default(),
            e0_profile_bank_limit_bytes: None,
            row_decomp_cache: HashMap::default(),
            basis_cache: if inherit_worker_caches {
                self.basis_cache.clone()
            } else {
                HashMap::default()
            },
            basis_index_cache: if inherit_worker_caches {
                self.basis_index_cache.clone()
            } else {
                HashMap::default()
            },
            algebra_basis_index_cache: if inherit_worker_caches {
                self.algebra_basis_index_cache.clone()
            } else {
                HashMap::default()
            },
            full_basis_lookup_cache: if inherit_worker_caches {
                self.full_basis_lookup_cache.clone()
            } else {
                HashMap::default()
            },
            coeff_signature_basis_cache: if inherit_worker_caches {
                self.coeff_signature_basis_cache.clone()
            } else {
                HashMap::default()
            },
            basis_signature_cache: if inherit_worker_caches {
                self.basis_signature_cache.clone()
            } else {
                HashMap::default()
            },
            basis_signature_index_cache: if inherit_worker_caches {
                self.basis_signature_index_cache.clone()
            } else {
                HashMap::default()
            },
            basis_signature_routing_cache: if inherit_worker_caches {
                self.basis_signature_routing_cache.clone()
            } else {
                HashMap::default()
            },
            basis_signature_to_zero_cache: if inherit_worker_caches {
                self.basis_signature_to_zero_cache.clone()
            } else {
                HashMap::default()
            },
            signature_matrix_cache: if inherit_worker_caches {
                self.signature_matrix_cache.clone()
            } else {
                HashMap::default()
            },
            signature_solver_cache: if inherit_worker_caches {
                self.signature_solver_cache.clone()
            } else {
                HashMap::default()
            },
            fixed_t_previous_task_micros: self.fixed_t_previous_task_micros.clone(),
            fixed_t_previous_group_ranges: self.fixed_t_previous_group_ranges.clone(),
            fixed_t_worker_product_caches: Vec::new(),
            cache_policy: self.cache_policy,
        }
    }

    // Worker inputs are borrowed or moved independently; bundling them would
    // add ownership plumbing without reducing work in this hot path.
    #[allow(clippy::too_many_arguments)]
    fn compute_isolated_bidegree_group(
        &self,
        group_index: usize,
        group: &[usize],
        t: usize,
        mode: &ComputeMode,
        basis_dims: &[usize],
        product_caches: Option<FixedTBatchProductCaches>,
        product_cache_limit_bytes: Option<usize>,
        e0_profile_bank_enabled: bool,
    ) -> Result<FixedTBatchGroupResult, String> {
        self.compute_isolated_bidegree_group_with_progress(
            group_index,
            group,
            t,
            mode,
            basis_dims,
            product_caches,
            product_cache_limit_bytes,
            e0_profile_bank_enabled,
            &|_: &str, _: String| Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_isolated_bidegree_group_with_progress<F>(
        &self,
        group_index: usize,
        group: &[usize],
        t: usize,
        mode: &ComputeMode,
        basis_dims: &[usize],
        product_caches: Option<FixedTBatchProductCaches>,
        product_cache_limit_bytes: Option<usize>,
        e0_profile_bank_enabled: bool,
        progress: &F,
    ) -> Result<FixedTBatchGroupResult, String>
    where
        F: Fn(&str, String) -> Result<(), String> + Sync,
    {
        let group_timer = Instant::now();
        let mut local = self.clone_with_shared_worker_caches();
        local.e0_profile_bank_limit_bytes = if e0_profile_bank_enabled {
            product_cache_limit_bytes
        } else {
            None
        };
        if let Some(product_caches) = product_caches {
            local.install_fixed_t_product_caches(product_caches);
        }
        let frozen_generator_count = local.generators.len();
        let mut results = Vec::with_capacity(group.len());
        for &s in group {
            let total_timer = Instant::now();
            let (d_domain_dim, d_target_dim, boundary_domain_dim, boundary_target_dim) =
                fixed_t_task_dimensions_from_dims(s, t, basis_dims);
            let candidate_subalgebras = local.candidate_subalgebra_names_for_mode(s, t, mode);
            let chosen_subalgebra = local
                .selected_algorithm2_subalgebra_for_mode(s, t, mode)
                .map(|subalgebra| subalgebra.name());
            let candidate_subalgebras_log = subalgebra_log_list(&candidate_subalgebras);
            let chosen_subalgebra_log =
                chosen_subalgebra.clone().unwrap_or_else(|| "-".to_string());
            let output_q = s + 1;
            let (reduced_domain_dim, reduced_target_dim, reduced_boundary_domain_dim) =
                local.reduced_task_dimensions_for_mode(s, t, mode);
            debug_assert_eq!(local.generators.len(), frozen_generator_count);
            progress(
                "worker_task_start",
                format!(
                    "local_group_index={group_index} s={s} output_q={output_q} chosen_subalgebra={chosen_subalgebra_log} candidate_subalgebras={candidate_subalgebras_log}{}",
                    process_memory_event_fields()
                ),
            )?;
            let compute_result = run_fixed_t_task_with_memory_samples(
                progress,
                group_index,
                s,
                &total_timer,
                || local.compute_step(s, t, mode),
            );
            let (path, added) = match compute_result {
                Ok(result) => result,
                Err(err) => {
                    let _ = progress(
                        "worker_task_error",
                        format!(
                            "local_group_index={group_index} s={s} output_q={output_q} chosen_subalgebra={chosen_subalgebra_log} candidate_subalgebras={candidate_subalgebras_log} error={}{}",
                            err.replace(' ', "_"),
                            process_memory_event_fields()
                        ),
                    );
                    return Err(err);
                }
            };
            let representatives = local.take_new_generator_representatives_and_truncate(
                frozen_generator_count,
                added,
                "isolated task",
                s,
                t,
            )?;
            let task = FixedTBatchTaskResult {
                s,
                output_q,
                t,
                path: format!("fixed-t-batch-isolated-{path}"),
                representatives,
                d_domain_dim,
                d_target_dim,
                boundary_domain_dim,
                boundary_target_dim,
                reduced_domain_dim,
                reduced_target_dim,
                reduced_boundary_domain_dim,
                candidate_subalgebras,
                chosen_subalgebra,
                matrix_s: 0.0,
                homology_s: 0.0,
                total_s: total_timer.elapsed().as_secs_f64(),
            };
            let task_wall_s = task.total_s;
            results.push(task);
            progress(
                "worker_task_done",
                format!(
                    "local_group_index={group_index} s={s} output_q={output_q} chosen_subalgebra={chosen_subalgebra_log} candidate_subalgebras={candidate_subalgebras_log} added={added} wall_s={:.6}{}",
                    task_wall_s,
                    process_memory_event_fields()
                ),
            )?;
        }
        local.log_fixed_t_worker_cache_profile("shared-arc", t, group);
        let product_caches = local.take_capped_fixed_t_product_caches(
            t,
            group,
            product_cache_limit_bytes,
            e0_profile_bank_enabled,
        );
        let caches = local.take_fixed_t_worker_caches_new_since(self);
        log_fixed_t_worker_cache_delta_profile(&caches, t, group);
        let task_sum_s = results.iter().map(|result| result.total_s).sum::<f64>();
        let task_max_s = results
            .iter()
            .map(|result| result.total_s)
            .fold(0.0, f64::max);
        let group_result = FixedTBatchGroupResult {
            group_index,
            group_first: group.first().copied().unwrap_or(0),
            group_last: group.last().copied().unwrap_or(0),
            group_len: group.len(),
            wall_s: group_timer.elapsed().as_secs_f64(),
            task_sum_s,
            task_max_s,
            tasks: results,
            caches,
            product_caches,
        };
        log_fixed_t_group_worker_done_profile(t, &group_result);
        Ok(group_result)
    }

    fn compute_isolated_bidegree_group_in_place(
        &mut self,
        group: &[usize],
        t: usize,
        mode: &ComputeMode,
        basis_dims: &[usize],
    ) -> Result<Vec<FixedTBatchTaskResult>, String> {
        self.compute_isolated_bidegree_group_in_place_with_progress(
            0,
            group,
            t,
            mode,
            basis_dims,
            &|_: &str, _: String| Ok(()),
        )
    }

    fn compute_isolated_bidegree_group_in_place_with_progress<F>(
        &mut self,
        group_index: usize,
        group: &[usize],
        t: usize,
        mode: &ComputeMode,
        basis_dims: &[usize],
        progress: &F,
    ) -> Result<Vec<FixedTBatchTaskResult>, String>
    where
        F: Fn(&str, String) -> Result<(), String> + Sync,
    {
        let frozen_generator_count = self.generators.len();
        let mut results = Vec::with_capacity(group.len());
        for &s in group {
            let total_timer = Instant::now();
            let (d_domain_dim, d_target_dim, boundary_domain_dim, boundary_target_dim) =
                fixed_t_task_dimensions_from_dims(s, t, basis_dims);
            let candidate_subalgebras = self.candidate_subalgebra_names_for_mode(s, t, mode);
            let chosen_subalgebra = self
                .selected_algorithm2_subalgebra_for_mode(s, t, mode)
                .map(|subalgebra| subalgebra.name());
            let candidate_subalgebras_log = subalgebra_log_list(&candidate_subalgebras);
            let chosen_subalgebra_log =
                chosen_subalgebra.clone().unwrap_or_else(|| "-".to_string());
            let output_q = s + 1;
            let (reduced_domain_dim, reduced_target_dim, reduced_boundary_domain_dim) =
                self.reduced_task_dimensions_for_mode(s, t, mode);
            debug_assert_eq!(self.generators.len(), frozen_generator_count);
            progress(
                "worker_task_start",
                format!(
                    "local_group_index={group_index} s={s} output_q={output_q} chosen_subalgebra={chosen_subalgebra_log} candidate_subalgebras={candidate_subalgebras_log}{}",
                    process_memory_event_fields()
                ),
            )?;
            let compute_result = run_fixed_t_task_with_memory_samples(
                progress,
                group_index,
                s,
                &total_timer,
                || self.compute_step(s, t, mode),
            );
            let (path, added) = match compute_result {
                Ok(result) => result,
                Err(err) => {
                    let _ = progress(
                        "worker_task_error",
                        format!(
                            "local_group_index={group_index} s={s} output_q={output_q} chosen_subalgebra={chosen_subalgebra_log} candidate_subalgebras={candidate_subalgebras_log} error={}{}",
                            err.replace(' ', "_"),
                            process_memory_event_fields()
                        ),
                    );
                    return Err(err);
                }
            };
            let representatives = self.take_new_generator_representatives_and_truncate(
                frozen_generator_count,
                added,
                "in-place isolated task",
                s,
                t,
            )?;
            results.push(FixedTBatchTaskResult {
                s,
                output_q,
                t,
                path: format!("fixed-t-batch-in-place-{path}"),
                representatives,
                d_domain_dim,
                d_target_dim,
                boundary_domain_dim,
                boundary_target_dim,
                reduced_domain_dim,
                reduced_target_dim,
                reduced_boundary_domain_dim,
                candidate_subalgebras,
                chosen_subalgebra,
                matrix_s: 0.0,
                homology_s: 0.0,
                total_s: total_timer.elapsed().as_secs_f64(),
            });
            progress(
                "worker_task_done",
                format!(
                    "local_group_index={group_index} s={s} output_q={output_q} chosen_subalgebra={chosen_subalgebra_log} candidate_subalgebras={candidate_subalgebras_log} added={added} wall_s={:.6}{}",
                    total_timer.elapsed().as_secs_f64(),
                    process_memory_event_fields()
                ),
            )?;
        }
        self.log_fixed_t_worker_cache_profile("in-place", t, group);
        Ok(results)
    }

    fn log_fixed_t_worker_cache_profile(&self, worker_cache: &str, t: usize, group: &[usize]) {
        if std::env::var_os("EXT_PROFILE_FIXED_T_WORKER_CACHE").is_none() {
            return;
        }
        let (mult_packed_terms, mult_packed_bytes) =
            estimate_vec_coeff_map(&self.mult_packed_cache);
        let (mult_e0_terms, mult_e0_bytes) = estimate_product_list_map(&self.mult_e0_cache);
        let mult_e0_empty_bytes = self.mult_e0_empty_cache.estimated_bytes();
        let mult_e0_profile_bank_bytes = estimate_e0_profile_bank_bytes(&self.mult_e0_profile_bank);
        let (row_decomp_rows, row_decomp_options, row_decomp_bytes) =
            estimate_row_decomp_cache(&self.row_decomp_cache);
        let matrix_bytes = self
            .signature_matrix_cache
            .values()
            .map(|matrix| matrix.estimated_bytes())
            .sum::<usize>();
        let solver_bytes = self
            .signature_solver_cache
            .values()
            .map(|solver| solver.estimated_bytes())
            .sum::<usize>();
        let group_first = group.first().copied().unwrap_or(0);
        let group_last = group.last().copied().unwrap_or(0);
        eprintln!(
            "fixed_t_batch_worker_cache t={t} worker_cache={worker_cache} group_first={group_first} group_last={group_last} group_len={} mult_packed={} mult_packed_terms={} mult_packed_mb={:.1} mult_e0_profile={:?} mult_e0={} mult_e0_terms={} mult_e0_mb={:.1} mult_e0_empty={} mult_e0_empty_segments={} mult_e0_empty_mb={:.1} e0_profile_bank={} e0_profile_bank_mb={:.1} mult_e0_total_mb={:.1} row_decomp={} row_decomp_rows={} row_decomp_options={} row_decomp_mb={:.1} sig_matrices={} sig_matrix_mb={:.1} sig_solvers={} sig_solver_mb={:.1}",
            group.len(),
            self.mult_packed_cache.len(),
            mult_packed_terms,
            bytes_to_mb(mult_packed_bytes),
            self.mult_e0_cache_profile,
            self.mult_e0_cache.len(),
            mult_e0_terms,
            bytes_to_mb(mult_e0_bytes),
            self.mult_e0_empty_cache.len(),
            self.mult_e0_empty_cache.segment_count(),
            bytes_to_mb(mult_e0_empty_bytes),
            self.mult_e0_profile_bank.len(),
            bytes_to_mb(mult_e0_profile_bank_bytes),
            bytes_to_mb(mult_e0_bytes + mult_e0_empty_bytes + mult_e0_profile_bank_bytes),
            self.row_decomp_cache.len(),
            row_decomp_rows,
            row_decomp_options,
            bytes_to_mb(row_decomp_bytes),
            self.signature_matrix_cache.len(),
            bytes_to_mb(matrix_bytes),
            self.signature_solver_cache.len(),
            bytes_to_mb(solver_bytes),
        );
    }

    fn take_capped_fixed_t_product_caches(
        &mut self,
        t: usize,
        group: &[usize],
        limit_bytes: Option<usize>,
        e0_profile_bank_enabled: bool,
    ) -> FixedTBatchProductCaches {
        let Some(limit_bytes) = limit_bytes else {
            return FixedTBatchProductCaches::default();
        };
        let mut caches = self.take_fixed_t_product_caches(e0_profile_bank_enabled);
        let fits_after_trim = if e0_profile_bank_enabled {
            caches.trim_to_limit(limit_bytes, self.cache_policy.e0_empty_cache)
        } else {
            caches.estimated_bytes() <= limit_bytes
        };
        let estimated_bytes = caches.estimated_bytes();
        let keep = fits_after_trim && estimated_bytes <= limit_bytes;
        if std::env::var_os("EXT_PROFILE_FIXED_T_WORKER_CACHE").is_some() {
            eprintln!(
                "fixed_t_worker_product_cache_persist t={t} group_first={} group_last={} group_len={} keep={} estimated_mb={:.1} limit_mb={:.1} mult_packed={} mult_e0={} mult_e0_empty={} e0_profile_bank={} e0_profile_bank_mb={:.1} row_decomp={}",
                group.first().copied().unwrap_or(0),
                group.last().copied().unwrap_or(0),
                group.len(),
                keep,
                bytes_to_mb(estimated_bytes),
                bytes_to_mb(limit_bytes),
                caches.mult_packed_cache.len(),
                caches.mult_e0_cache.len(),
                caches.mult_e0_empty_cache.len(),
                caches.e0_profile_bank_len(),
                bytes_to_mb(caches.e0_profile_bank_bytes()),
                caches.row_decomp_cache.len(),
            );
        }
        if keep {
            caches
        } else {
            FixedTBatchProductCaches::default()
        }
    }

    fn log_persistent_fixed_t_product_cache_profile(&self, t: usize) {
        if std::env::var_os("EXT_PROFILE_FIXED_T_WORKER_CACHE").is_none() {
            return;
        }
        let mut total_bytes = 0usize;
        let mut max_worker_bytes = 0usize;
        let mut kept_workers = 0usize;
        for persisted in &self.fixed_t_worker_product_caches {
            let bytes = persisted.caches.estimated_bytes();
            total_bytes = total_bytes.saturating_add(bytes);
            max_worker_bytes = max_worker_bytes.max(bytes);
            if !persisted.caches.is_empty() {
                kept_workers += 1;
            }
        }
        eprintln!(
            "fixed_t_persistent_worker_product_cache t={t} workers={} kept_workers={} total_mb={:.1} max_worker_mb={:.1}",
            self.fixed_t_worker_product_caches.len(),
            kept_workers,
            bytes_to_mb(total_bytes),
            bytes_to_mb(max_worker_bytes),
        );
    }

    fn take_fixed_t_worker_caches(self) -> FixedTBatchWorkerCaches {
        FixedTBatchWorkerCaches {
            basis_cache: self.basis_cache,
            basis_index_cache: self.basis_index_cache,
            algebra_basis_index_cache: self.algebra_basis_index_cache,
            full_basis_lookup_cache: self.full_basis_lookup_cache,
            coeff_signature_basis_cache: self.coeff_signature_basis_cache,
            basis_signature_cache: self.basis_signature_cache,
            basis_signature_index_cache: self.basis_signature_index_cache,
            basis_signature_routing_cache: self.basis_signature_routing_cache,
            basis_signature_to_zero_cache: self.basis_signature_to_zero_cache,
            signature_matrix_cache: self.signature_matrix_cache,
            signature_solver_cache: self.signature_solver_cache,
        }
    }

    fn take_fixed_t_worker_caches_new_since(self, baseline: &Self) -> FixedTBatchWorkerCaches {
        let mut caches = self.take_fixed_t_worker_caches();
        filter_fixed_t_worker_caches_new_since(&mut caches, baseline);
        caches
    }

    fn take_fixed_t_product_caches(
        &mut self,
        e0_profile_bank_enabled: bool,
    ) -> FixedTBatchProductCaches {
        if e0_profile_bank_enabled {
            self.shelve_active_e0_profile_cache();
            self.trim_e0_profile_bank_to_limit();
        } else {
            self.mult_e0_profile_bank.clear();
        }
        let mut replacement_empty = EmptyProductCache::default();
        replacement_empty.set_policy(self.cache_policy.e0_empty_cache);
        FixedTBatchProductCaches {
            mult_packed_cache: std::mem::take(&mut self.mult_packed_cache),
            mult_e0_cache: std::mem::take(&mut self.mult_e0_cache),
            mult_e0_empty_cache: std::mem::replace(
                &mut self.mult_e0_empty_cache,
                replacement_empty,
            ),
            mult_e0_cache_profile: self.mult_e0_cache_profile.take(),
            mult_e0_profile_bank: std::mem::take(&mut self.mult_e0_profile_bank),
            row_decomp_cache: std::mem::take(&mut self.row_decomp_cache),
        }
    }

    fn install_fixed_t_product_caches(&mut self, mut caches: FixedTBatchProductCaches) {
        caches.set_policy(self.cache_policy.e0_empty_cache);
        self.mult_packed_cache = caches.mult_packed_cache;
        self.mult_e0_cache = caches.mult_e0_cache;
        self.mult_e0_empty_cache = caches.mult_e0_empty_cache;
        self.mult_e0_cache_profile = caches.mult_e0_cache_profile;
        self.mult_e0_profile_bank = caches.mult_e0_profile_bank;
        self.row_decomp_cache = caches.row_decomp_cache;
    }

    fn take_persistent_fixed_t_product_caches(
        &mut self,
        groups: &[Vec<usize>],
        enabled: bool,
    ) -> Vec<Option<FixedTBatchProductCaches>> {
        if !enabled {
            self.fixed_t_worker_product_caches.clear();
            return (0..groups.len()).map(|_| None).collect();
        }

        let mut stored = std::mem::take(&mut self.fixed_t_worker_product_caches)
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
        let mut out = Vec::with_capacity(groups.len());
        for group in groups {
            let mut caches = take_best_persistent_product_cache_for_group(&mut stored, group)
                .map(|persisted| persisted.caches)
                .unwrap_or_default();
            caches.set_policy(self.cache_policy.e0_empty_cache);
            out.push(Some(caches));
        }
        out
    }

    fn merge_fixed_t_worker_caches(&mut self, caches: FixedTBatchWorkerCaches) {
        self.basis_cache.extend(caches.basis_cache);
        self.basis_index_cache.extend(caches.basis_index_cache);
        self.algebra_basis_index_cache
            .extend(caches.algebra_basis_index_cache);
        self.full_basis_lookup_cache
            .extend(caches.full_basis_lookup_cache);
        self.coeff_signature_basis_cache
            .extend(caches.coeff_signature_basis_cache);
        self.basis_signature_cache
            .extend(caches.basis_signature_cache);
        self.basis_signature_index_cache
            .extend(caches.basis_signature_index_cache);
        self.basis_signature_routing_cache
            .extend(caches.basis_signature_routing_cache);
        self.basis_signature_to_zero_cache
            .extend(caches.basis_signature_to_zero_cache);
        self.signature_matrix_cache
            .extend(caches.signature_matrix_cache);
        self.signature_solver_cache
            .extend(caches.signature_solver_cache);
    }

    fn take_new_generator_representatives_and_truncate(
        &mut self,
        len: usize,
        added: usize,
        label: &str,
        s: usize,
        t: usize,
    ) -> Result<Vec<Vec<ModuleTerm>>, String> {
        if self.generators.len() < len {
            return Err(format!(
                "internal error: {label} (s,t)=({s},{t}) rollback target {len} exceeds generator count {}",
                self.generators.len()
            ));
        }
        let actual_added = self.generators.len() - len;
        if actual_added != added {
            return Err(format!(
                "internal error: {label} (s,t)=({s},{t}) reported added={added} but appended {actual_added} generators"
            ));
        }

        let mut representatives = Vec::with_capacity(actual_added);
        while self.generators.len() > len {
            let generator = self
                .generators
                .pop()
                .ok_or_else(|| "internal error: generator stack underflow".to_string())?;
            let Some(gens) = self.gens_by_s.get_mut(generator.s) else {
                return Err(format!(
                    "internal error: gens_by_s missing homological degree {} during rollback",
                    generator.s
                ));
            };
            match gens.pop() {
                Some(id) if id == generator.id => {}
                Some(id) => {
                    return Err(format!(
                        "internal error: rollback expected generator {}, found {}",
                        generator.id, id
                    ));
                }
                None => {
                    return Err(format!(
                        "internal error: empty gens_by_s[{}] during rollback",
                        generator.s
                    ));
                }
            }
            representatives.push(
                Arc::try_unwrap(generator.differential)
                    .unwrap_or_else(|differential| differential.as_ref().clone()),
            );
        }
        representatives.reverse();
        Ok(representatives)
    }

    pub fn from_snapshot(_max_t: usize, snapshot: ResolutionSnapshot) -> Result<Self, String> {
        if snapshot.generators.is_empty() {
            return Err("checkpoint contains no generators".to_string());
        }

        let mut out = Self::empty();
        for (expected_id, generator) in snapshot.generators.into_iter().enumerate() {
            if generator.id != expected_id {
                return Err(format!(
                    "checkpoint generator id mismatch: expected {expected_id}, found {}",
                    generator.id
                ));
            }
            if generator.id == 0
                && (generator.s != 0 || generator.t != 0 || !generator.differential.is_empty())
            {
                return Err("checkpoint generator 0 is not the sphere generator".to_string());
            }
            for term in generator.differential.iter() {
                if term.generator >= expected_id {
                    return Err(format!(
                        "checkpoint differential for generator {} references non-earlier generator {}",
                        generator.id, term.generator
                    ));
                }
                let target = &out.generators[term.generator];
                if target.s + 1 != generator.s {
                    return Err(format!(
                        "checkpoint differential for generator {} has target generator {} in s={}, expected s={}",
                        generator.id,
                        term.generator,
                        target.s,
                        generator.s.saturating_sub(1)
                    ));
                }
                let coeff_degree = Milnor::from_packed(term.coeff_packed).degree();
                if target.t + coeff_degree != generator.t {
                    return Err(format!(
                        "checkpoint degree mismatch in d(g{}): coefficient degree {} plus target t={} does not equal source t={}",
                        generator.id, coeff_degree, target.t, generator.t
                    ));
                }
                if coeff_degree == 0 {
                    return Err(format!(
                        "checkpoint minimality violation in d(g{}): unit coefficient on g{}",
                        generator.id, term.generator
                    ));
                }
            }
            let differential = Arc::try_unwrap(generator.differential)
                .unwrap_or_else(|shared| shared.as_ref().clone());
            let id = out.add_generator(generator.s, generator.t, differential);
            if id != expected_id {
                return Err(format!(
                    "checkpoint reconstruction assigned generator id {id}, expected {expected_id}"
                ));
            }
        }
        Ok(out)
    }

    pub fn from_sparse_snapshot(
        _max_t: usize,
        total_generator_count: usize,
        snapshot: ResolutionSnapshot,
    ) -> Result<Self, String> {
        if total_generator_count == 0 {
            return Err("sparse checkpoint contains no generator id space".to_string());
        }

        let mut out = Self::empty();
        let empty_differential = Arc::new(Vec::new());
        out.generators = (0..total_generator_count)
            .map(|id| Generator {
                id,
                s: usize::MAX,
                t: 0,
                differential: Arc::clone(&empty_differential),
            })
            .collect();
        let mut loaded = vec![false; total_generator_count];
        if total_generator_count > 0 {
            out.generators[0] = Generator {
                id: 0,
                s: 0,
                t: 0,
                differential: Arc::new(Vec::new()),
            };
            loaded[0] = true;
        }

        let mut generators = snapshot.generators;
        generators.sort_by_key(|generator| generator.id);
        for generator in generators {
            if generator.id >= total_generator_count {
                return Err(format!(
                    "sparse checkpoint generator id {} is outside total_generator_count={total_generator_count}",
                    generator.id
                ));
            }
            if loaded[generator.id] && generator.id != 0 {
                return Err(format!(
                    "sparse checkpoint loaded duplicate generator id {}",
                    generator.id
                ));
            }
            if generator.id == 0
                && (generator.s != 0 || generator.t != 0 || !generator.differential.is_empty())
            {
                return Err("sparse checkpoint generator 0 is not the sphere generator".to_string());
            }
            for term in generator.differential.iter() {
                if term.generator >= generator.id {
                    return Err(format!(
                        "sparse checkpoint differential for generator {} references non-earlier generator {}",
                        generator.id, term.generator
                    ));
                }
                if !loaded.get(term.generator).copied().unwrap_or(false) {
                    return Err(format!(
                        "sparse checkpoint differential for generator {} needs unloaded target generator {}",
                        generator.id, term.generator
                    ));
                }
                let target = &out.generators[term.generator];
                if target.s + 1 != generator.s {
                    return Err(format!(
                        "sparse checkpoint differential for generator {} has target generator {} in s={}, expected s={}",
                        generator.id,
                        term.generator,
                        target.s,
                        generator.s.saturating_sub(1)
                    ));
                }
                let coeff_degree = Milnor::from_packed(term.coeff_packed).degree();
                if target.t + coeff_degree != generator.t {
                    return Err(format!(
                        "sparse checkpoint degree mismatch in d(g{}): coefficient degree {} plus target t={} does not equal source t={}",
                        generator.id, coeff_degree, target.t, generator.t
                    ));
                }
                if coeff_degree == 0 {
                    return Err(format!(
                        "sparse checkpoint minimality violation in d(g{}): unit coefficient on g{}",
                        generator.id, term.generator
                    ));
                }
            }
            let s = generator.s;
            while out.gens_by_s.len() <= s {
                out.gens_by_s.push(Vec::new());
            }
            out.gens_by_s[s].push(generator.id);
            let id = generator.id;
            out.generators[id] = generator;
            loaded[id] = true;
        }
        for gens in &mut out.gens_by_s {
            gens.sort_unstable();
        }
        Ok(out)
    }

    #[cfg(test)]
    pub fn merge_sparse_snapshot(
        &mut self,
        total_generator_count: usize,
        snapshot: ResolutionSnapshot,
    ) -> Result<(), String> {
        if total_generator_count < self.generators.len() {
            return Err(format!(
                "sparse merge total_generator_count={total_generator_count} is below current generator space {}",
                self.generators.len()
            ));
        }
        if self.generators.is_empty() {
            return Err("sparse merge requires an initialized resolution".to_string());
        }

        let empty_differential = Arc::new(Vec::new());
        while self.generators.len() < total_generator_count {
            let id = self.generators.len();
            self.generators.push(Generator {
                id,
                s: usize::MAX,
                t: 0,
                differential: Arc::clone(&empty_differential),
            });
        }

        let mut generators = snapshot.generators;
        generators.sort_by_key(|generator| generator.id);
        for generator in generators {
            if generator.id >= total_generator_count {
                return Err(format!(
                    "sparse merge generator id {} is outside total_generator_count={total_generator_count}",
                    generator.id
                ));
            }
            if generator.id == 0
                && (generator.s != 0 || generator.t != 0 || !generator.differential.is_empty())
            {
                return Err("sparse merge generator 0 is not the sphere generator".to_string());
            }
            if self.generators[generator.id].s != usize::MAX {
                if generator.id == 0 {
                    continue;
                }
                return Err(format!(
                    "sparse merge would replace already-loaded generator id {}",
                    generator.id
                ));
            }
            for term in generator.differential.iter() {
                if term.generator >= generator.id {
                    return Err(format!(
                        "sparse merge differential for generator {} references non-earlier generator {}",
                        generator.id, term.generator
                    ));
                }
                let Some(target) = self.generators.get(term.generator) else {
                    return Err(format!(
                        "sparse merge differential for generator {} references missing target {}",
                        generator.id, term.generator
                    ));
                };
                if target.s == usize::MAX {
                    return Err(format!(
                        "sparse merge differential for generator {} needs unloaded target generator {}",
                        generator.id, term.generator
                    ));
                }
                if target.s + 1 != generator.s {
                    return Err(format!(
                        "sparse merge differential for generator {} has target generator {} in s={}, expected s={}",
                        generator.id,
                        term.generator,
                        target.s,
                        generator.s.saturating_sub(1)
                    ));
                }
                let coeff_degree = Milnor::from_packed(term.coeff_packed).degree();
                if target.t + coeff_degree != generator.t {
                    return Err(format!(
                        "sparse merge degree mismatch in d(g{}): coefficient degree {} plus target t={} does not equal source t={}",
                        generator.id, coeff_degree, target.t, generator.t
                    ));
                }
                if coeff_degree == 0 {
                    return Err(format!(
                        "sparse merge minimality violation in d(g{}): unit coefficient on g{}",
                        generator.id, term.generator
                    ));
                }
            }
            let mut differential = generator.differential.as_ref().clone();
            sort_terms(&mut differential);
            let s = generator.s;
            while self.gens_by_s.len() <= s {
                self.gens_by_s.push(Vec::new());
            }
            self.gens_by_s[s].push(generator.id);
            self.generators[generator.id] = Generator {
                id: generator.id,
                s,
                t: generator.t,
                differential: Arc::new(differential),
            };
            self.invalidate_basis_caches(s, generator.t);
        }
        for gens in &mut self.gens_by_s {
            gens.sort_unstable();
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn retain_sparse_homological_degrees_from(
        &mut self,
        min_generator_id: usize,
        meta_s: &BTreeSet<usize>,
        diff_s: &BTreeSet<usize>,
    ) -> Result<(usize, usize), String> {
        if min_generator_id > self.generators.len() {
            return Err(format!(
                "sparse retain min_generator_id={min_generator_id} exceeds generator space {}",
                self.generators.len()
            ));
        }

        let empty_differential = Arc::new(Vec::new());
        let mut removed_by_s: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut changed_degrees = Vec::new();
        let mut removed = 0_usize;
        let mut stripped_differentials = 0_usize;

        for id in min_generator_id..self.generators.len() {
            let generator = &mut self.generators[id];
            if generator.s == usize::MAX {
                continue;
            }
            let s = generator.s;
            let t = generator.t;
            if !meta_s.contains(&s) {
                removed_by_s.entry(s).or_default().push(id);
                *generator = Generator {
                    id,
                    s: usize::MAX,
                    t: 0,
                    differential: Arc::clone(&empty_differential),
                };
                changed_degrees.push((s, t));
                removed += 1;
            } else if !diff_s.contains(&s) && !generator.differential.is_empty() {
                generator.differential = Arc::clone(&empty_differential);
                changed_degrees.push((s, t));
                stripped_differentials += 1;
            }
        }

        for (s, ids) in removed_by_s {
            let Some(gens) = self.gens_by_s.get_mut(s) else {
                return Err(format!(
                    "sparse retain missing gens_by_s[{s}] while removing {} generators",
                    ids.len()
                ));
            };
            let remove_ids = ids.into_iter().collect::<FastHashSet<_>>();
            let before = gens.len();
            gens.retain(|id| !remove_ids.contains(id));
            if before - gens.len() != remove_ids.len() {
                return Err(format!(
                    "sparse retain removed {} generators from gens_by_s[{s}], expected {}",
                    before - gens.len(),
                    remove_ids.len()
                ));
            }
        }

        changed_degrees.sort_unstable();
        changed_degrees.dedup();
        for (s, t) in changed_degrees {
            self.invalidate_basis_caches(s, t);
        }
        Ok((removed, stripped_differentials))
    }

    pub fn generator_count(&self) -> usize {
        self.generators.len()
    }

    pub fn generators(&self) -> &[Generator] {
        &self.generators
    }

    fn estimated_resolution_bytes(&self) -> usize {
        let algebra_basis_bytes = self
            .algebra_basis
            .iter()
            .map(|basis| basis.capacity() * std::mem::size_of::<CoeffKey>())
            .sum::<usize>()
            + self.algebra_basis.capacity() * std::mem::size_of::<Vec<CoeffKey>>();
        let generator_bytes = self.generators.capacity() * std::mem::size_of::<Generator>()
            + self
                .generators
                .iter()
                .map(|generator| {
                    generator.differential.capacity() * std::mem::size_of::<ModuleTerm>()
                })
                .sum::<usize>();
        let gens_by_s_bytes = self.gens_by_s.capacity() * std::mem::size_of::<Vec<usize>>()
            + self
                .gens_by_s
                .iter()
                .map(|ids| ids.capacity() * std::mem::size_of::<usize>())
                .sum::<usize>();
        algebra_basis_bytes + generator_bytes + gens_by_s_bytes
    }

    fn log_memory_cache_estimate(&self, phase: &str, s: usize, t: usize) {
        if std::env::var_os("EXT_PROFILE_MEMORY").is_none() {
            return;
        }

        let matrix_bytes = self
            .signature_matrix_cache
            .values()
            .map(|matrix| matrix.estimated_bytes())
            .sum::<usize>();
        let matrix_columns = self
            .signature_matrix_cache
            .values()
            .map(|matrix| matrix.columns.len())
            .sum::<usize>();
        let solver_bytes = self
            .signature_solver_cache
            .values()
            .map(|solver| solver.estimated_bytes())
            .sum::<usize>();
        let solver_pivots = self
            .signature_solver_cache
            .values()
            .map(|solver| solver.pivot_count())
            .sum::<usize>();
        let signature_basis_bytes = self
            .basis_signature_cache
            .values()
            .map(|basis| basis.capacity() * std::mem::size_of::<BasisElem>())
            .sum::<usize>();
        let full_lookup_bytes = self
            .full_basis_lookup_cache
            .values()
            .map(|lookup| lookup.estimated_bytes())
            .sum::<usize>();
        let (mult_packed_terms, mult_packed_bytes) =
            estimate_vec_coeff_map(&self.mult_packed_cache);
        let (mult_e0_terms, mult_e0_bytes) = estimate_product_list_map(&self.mult_e0_cache);
        let mult_e0_empty_bytes = self.mult_e0_empty_cache.estimated_bytes();
        let mult_e0_profile_bank_bytes = estimate_e0_profile_bank_bytes(&self.mult_e0_profile_bank);
        let mult_e0_total_bytes = mult_e0_bytes + mult_e0_empty_bytes + mult_e0_profile_bank_bytes;
        let (row_decomp_rows, row_decomp_options, row_decomp_bytes) =
            estimate_row_decomp_cache(&self.row_decomp_cache);
        let resolution_bytes = self.estimated_resolution_bytes();
        let basis_bytes = estimate_arc_vec_cache(&self.basis_cache);
        let basis_index_bytes = estimate_arc_map_cache(&self.basis_index_cache);
        let algebra_index_bytes = estimate_arc_map_cache(&self.algebra_basis_index_cache);
        let coeff_signature_bytes = estimate_arc_vec_cache(&self.coeff_signature_basis_cache);
        let signature_index_bytes = estimate_arc_map_cache(&self.basis_signature_index_cache);
        let signature_routing_bytes = estimate_arc_vec_cache(&self.basis_signature_routing_cache);
        let signature_translation_bytes = self
            .basis_signature_to_zero_cache
            .values()
            .map(|translation| {
                std::mem::size_of::<SignatureTranslation>()
                    + translation.to_zero.capacity() * std::mem::size_of::<usize>()
                    + translation.from_zero.capacity() * std::mem::size_of::<usize>()
            })
            .sum::<usize>();

        eprintln!(
            "ext_mem_cache phase={phase} t={t} s={s} sig_matrices={} sig_matrix_cols={} sig_matrix_mb={:.1} sig_solvers={} sig_solver_pivots={} sig_solver_mb={:.1} sig_basis={} sig_basis_mb={:.1} full_lookup={} full_lookup_mb={:.1}",
            self.signature_matrix_cache.len(),
            matrix_columns,
            bytes_to_mb(matrix_bytes),
            self.signature_solver_cache.len(),
            solver_pivots,
            bytes_to_mb(solver_bytes),
            self.basis_signature_cache.len(),
            bytes_to_mb(signature_basis_bytes),
            self.full_basis_lookup_cache.len(),
            bytes_to_mb(full_lookup_bytes),
        );
        eprintln!(
            "ext_mem_product phase={phase} t={t} s={s} mult_packed={} mult_packed_terms={} mult_packed_mb={:.1} mult_e0={} mult_e0_terms={} mult_e0_mb={:.1} mult_e0_empty={} mult_e0_empty_segments={} mult_e0_empty_mb={:.1} e0_profile_bank={} e0_profile_bank_mb={:.1} mult_e0_total_mb={:.1} row_decomp={} row_decomp_rows={} row_decomp_options={} row_decomp_mb={:.1}",
            self.mult_packed_cache.len(),
            mult_packed_terms,
            bytes_to_mb(mult_packed_bytes),
            self.mult_e0_cache.len(),
            mult_e0_terms,
            bytes_to_mb(mult_e0_bytes),
            self.mult_e0_empty_cache.len(),
            self.mult_e0_empty_cache.segment_count(),
            bytes_to_mb(mult_e0_empty_bytes),
            self.mult_e0_profile_bank.len(),
            bytes_to_mb(mult_e0_profile_bank_bytes),
            bytes_to_mb(mult_e0_total_bytes),
            self.row_decomp_cache.len(),
            row_decomp_rows,
            row_decomp_options,
            bytes_to_mb(row_decomp_bytes),
        );
        eprintln!(
            "ext_mem_basis phase={phase} t={t} s={s} resolution_mb={:.1} basis_cache={} basis_mb={:.1} basis_index_cache={} basis_index_mb={:.1} algebra_index_cache={} algebra_index_mb={:.1} coeff_signature_cache={} coeff_signature_mb={:.1} sig_index_cache={} sig_index_mb={:.1} sig_routing_cache={} sig_routing_mb={:.1} sig_translation_cache={} sig_translation_mb={:.1}",
            bytes_to_mb(resolution_bytes),
            self.basis_cache.len(),
            bytes_to_mb(basis_bytes),
            self.basis_index_cache.len(),
            bytes_to_mb(basis_index_bytes),
            self.algebra_basis_index_cache.len(),
            bytes_to_mb(algebra_index_bytes),
            self.coeff_signature_basis_cache.len(),
            bytes_to_mb(coeff_signature_bytes),
            self.basis_signature_index_cache.len(),
            bytes_to_mb(signature_index_bytes),
            self.basis_signature_routing_cache.len(),
            bytes_to_mb(signature_routing_bytes),
            self.basis_signature_to_zero_cache.len(),
            bytes_to_mb(signature_translation_bytes),
        );
    }

    fn clear_e0_product_cache(&mut self, phase: &str, s: usize, t: usize) {
        if self.mult_e0_cache.is_empty()
            && self.mult_e0_empty_cache.is_empty()
            && self.mult_e0_profile_bank.is_empty()
        {
            self.mult_e0_cache_profile = None;
            return;
        }
        if self.cache_policy.e0_cache_scope == E0CacheScope::Profile {
            return;
        }
        let entries = self.mult_e0_cache.len();
        let capacity = self.mult_e0_cache.capacity();
        let empty_entries = self.mult_e0_empty_cache.len();
        let empty_capacity = self.mult_e0_empty_cache.capacity_items();
        let empty_segments = self.mult_e0_empty_cache.segment_count();
        let bank_profiles = self.mult_e0_profile_bank.len();
        let bank_bytes = estimate_e0_profile_bank_bytes(&self.mult_e0_profile_bank);
        self.mult_e0_cache.clear();
        self.mult_e0_cache.shrink_to_fit();
        self.mult_e0_empty_cache.clear_and_shrink();
        self.mult_e0_profile_bank.clear();
        self.e0_profile_bank_limit_bytes = None;
        self.mult_e0_cache_profile = None;
        release_allocator_pressure();
        if std::env::var_os("EXT_PROFILE_MEMORY").is_some() {
            eprintln!(
                "ext_mem_product_clear phase={phase} t={t} s={s} cleared_entries={} old_capacity={} cleared_empty_entries={} old_empty_capacity={} old_empty_segments={} cleared_bank_profiles={} cleared_bank_mb={:.1}",
                entries,
                capacity,
                empty_entries,
                empty_capacity,
                empty_segments,
                bank_profiles,
                bytes_to_mb(bank_bytes),
            );
        }
    }

    fn filter_cached_e0_empty_product_bounds(&self, missing: &mut Vec<ProductKeyWithBound>) {
        missing.retain(|&(left_key, right_key, _)| {
            !self.mult_e0_empty_cache.contains(&(left_key, right_key))
        });
    }

    fn should_prefilter_cached_e0_empty_products(&self, missing_len: usize) -> bool {
        missing_len != 0 && !self.mult_e0_empty_cache.is_empty()
    }

    fn merge_e0_empty_products(&mut self, new_empty: Vec<(CoeffKey, CoeffKey)>) {
        self.mult_e0_empty_cache.insert_sorted(new_empty);
    }

    fn activate_e0_product_profile(&mut self, profile_key: E0ProfileKey) {
        if self.mult_e0_cache_profile == Some(profile_key) {
            return;
        }

        if self.e0_profile_bank_limit_bytes.is_some() {
            self.shelve_active_e0_profile_cache();
            if let Some(mut caches) = self.mult_e0_profile_bank.remove(&profile_key) {
                caches.set_policy(self.cache_policy.e0_empty_cache);
                self.mult_e0_cache = caches.mult_e0_cache;
                self.mult_e0_empty_cache = caches.mult_e0_empty_cache;
            } else {
                self.mult_e0_cache.clear();
                self.mult_e0_cache.shrink_to_fit();
                self.mult_e0_empty_cache.clear_and_shrink();
            }
            self.mult_e0_cache_profile = Some(profile_key);
            self.trim_e0_profile_bank_to_limit();
        } else {
            self.mult_e0_cache.clear();
            self.mult_e0_cache.shrink_to_fit();
            self.mult_e0_empty_cache.clear_and_shrink();
            self.mult_e0_profile_bank.clear();
            self.mult_e0_cache_profile = Some(profile_key);
        }
    }

    fn shelve_active_e0_profile_cache(&mut self) {
        let Some(profile) = self.mult_e0_cache_profile.take() else {
            return;
        };
        let mut replacement_empty = EmptyProductCache::default();
        replacement_empty.set_policy(self.cache_policy.e0_empty_cache);
        let mut caches = E0ProfileProductCaches {
            mult_e0_cache: std::mem::take(&mut self.mult_e0_cache),
            mult_e0_empty_cache: std::mem::replace(
                &mut self.mult_e0_empty_cache,
                replacement_empty,
            ),
        };
        caches.set_policy(self.cache_policy.e0_empty_cache);
        if !caches.is_empty() {
            self.mult_e0_profile_bank.insert(profile, caches);
        }
    }

    fn active_e0_profile_cache_bytes(&self) -> usize {
        let (_, mult_e0_bytes) = estimate_product_list_map(&self.mult_e0_cache);
        mult_e0_bytes.saturating_add(self.mult_e0_empty_cache.estimated_bytes())
    }

    fn trim_e0_profile_bank_to_limit(&mut self) {
        let Some(limit_bytes) = self.e0_profile_bank_limit_bytes else {
            return;
        };
        let active_bytes = self.active_e0_profile_cache_bytes();
        if active_bytes >= limit_bytes {
            self.mult_e0_profile_bank.clear();
            return;
        }

        let mut bank_bytes = estimate_e0_profile_bank_bytes(&self.mult_e0_profile_bank);
        if active_bytes.saturating_add(bank_bytes) <= limit_bytes {
            return;
        }

        let mut profiles = self
            .mult_e0_profile_bank
            .iter()
            .map(|(&profile, caches)| (profile, caches.estimated_bytes()))
            .collect::<Vec<_>>();
        profiles.sort_by_key(|(_, bytes)| *bytes);
        for (profile, bytes) in profiles {
            if active_bytes.saturating_add(bank_bytes) <= limit_bytes {
                break;
            }
            if self.mult_e0_profile_bank.remove(&profile).is_some() {
                bank_bytes = bank_bytes.saturating_sub(bytes);
            }
        }
    }

    pub fn prune_t_caches(&mut self, next_t: usize, window: usize) {
        let keep_from = next_t.saturating_sub(window);
        self.basis_cache.retain(|&(_, t), _| t >= keep_from);
        self.basis_index_cache.retain(|&(_, t), _| t >= keep_from);
        self.full_basis_lookup_cache
            .retain(|&(_, t), _| t >= keep_from);
        self.basis_signature_cache
            .retain(|key, _| key.t >= keep_from);
        self.basis_signature_index_cache
            .retain(|key, _| key.t >= keep_from);
        self.basis_signature_routing_cache
            .retain(|key, _| key.t >= keep_from);
        self.basis_signature_to_zero_cache
            .retain(|key, _| key.t >= keep_from);
        self.signature_matrix_cache
            .retain(|key, _| key.t >= keep_from);
        self.signature_solver_cache
            .retain(|key, _| key.t >= keep_from);
    }

    fn choose_subalgebra<'a>(
        &mut self,
        s: usize,
        t: usize,
        subalgebras: &'a [Subalgebra],
        force: bool,
    ) -> Option<&'a Subalgebra> {
        // The selection condition is for the fixed-t homology task H_s(D^(t)).
        // The generators produced by this task live in homological degree s+1.
        let mut best = None::<((usize, usize), &'a Subalgebra)>;
        for subalgebra in subalgebras {
            if subalgebra_selection_is_disabled(subalgebra) {
                continue;
            }
            if !(force || subalgebra.selection_condition_applies(s, t)) {
                continue;
            }
            let current = self.basis_signature_dim(s, t, subalgebra, 0);
            let prev = s
                .checked_sub(1)
                .map(|prev_s| self.basis_signature_dim(prev_s, t, subalgebra, 0))
                .unwrap_or(0);
            let next = self.basis_signature_dim(s + 1, t, subalgebra, 0);
            let key = (
                current.saturating_add(prev).saturating_add(next),
                subalgebra.selection_priority(),
            );
            if best
                .as_ref()
                .map(|(best_key, best_subalgebra)| {
                    candidate_subalgebra_is_better(subalgebra, key, best_subalgebra, *best_key)
                })
                .unwrap_or(true)
            {
                best = Some((key, subalgebra));
            }
        }
        best.map(|(_, subalgebra)| subalgebra)
    }

    fn selected_algorithm2_subalgebra_for_mode(
        &mut self,
        s: usize,
        t: usize,
        mode: &ComputeMode,
    ) -> Option<Subalgebra> {
        match mode {
            ComputeMode::Auto {
                subalgebras, force, ..
            }
            | ComputeMode::AutoBoundedNaiveFallback { subalgebras, force } => {
                self.choose_subalgebra(s, t, subalgebras, *force).cloned()
            }
            ComputeMode::Accelerated {
                subalgebra, force, ..
            } if !subalgebra_selection_is_disabled(subalgebra)
                && (*force || subalgebra.selection_condition_applies(s, t)) =>
            {
                Some(subalgebra.clone())
            }
            _ => None,
        }
    }

    fn candidate_subalgebra_names_for_mode(
        &self,
        s: usize,
        t: usize,
        mode: &ComputeMode,
    ) -> Vec<String> {
        match mode {
            ComputeMode::Auto {
                subalgebras, force, ..
            }
            | ComputeMode::AutoBoundedNaiveFallback { subalgebras, force } => subalgebras
                .iter()
                .filter(|subalgebra| {
                    !subalgebra_selection_is_disabled(subalgebra)
                        && (*force || subalgebra.selection_condition_applies(s, t))
                })
                .map(Subalgebra::name)
                .collect(),
            ComputeMode::Accelerated {
                subalgebra, force, ..
            } if !subalgebra_selection_is_disabled(subalgebra)
                && (*force || subalgebra.selection_condition_applies(s, t)) =>
            {
                vec![subalgebra.name()]
            }
            _ => Vec::new(),
        }
    }

    fn reduced_task_dimensions_for_mode(
        &mut self,
        s: usize,
        t: usize,
        mode: &ComputeMode,
    ) -> (Option<usize>, Option<usize>, Option<usize>) {
        let Some(subalgebra) = self.selected_algorithm2_subalgebra_for_mode(s, t, mode) else {
            return (None, None, None);
        };
        let domain = self.basis_signature_dim(s, t, &subalgebra, 0);
        let target = if s == 0 {
            usize::from(t == 0)
        } else {
            self.basis_signature_dim(s - 1, t, &subalgebra, 0)
        };
        let boundary_domain = self.basis_signature_dim(s + 1, t, &subalgebra, 0);
        (Some(domain), Some(target), Some(boundary_domain))
    }

    fn step_naive(&mut self, s: usize, t: usize) -> Result<usize, String> {
        let reps = self.naive_homology_representatives(s, t)?;
        let added = reps.len();
        for rep in reps {
            let diff = self.vector_to_terms(s, t, &rep);
            self.add_generator(s + 1, t, diff);
        }
        Ok(added)
    }

    fn step_algorithm2(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
    ) -> Result<usize, String> {
        let profile_detail = std::env::var_os("EXT_PROFILE_SIGNATURE_DETAIL").is_some();
        let profile_verbose = std::env::var_os("EXT_PROFILE_MEMORY_VERBOSE").is_some();
        let total_timer = Instant::now();
        let profile_label = format!("t={t} s={s} B={}", subalgebra.name());
        log_process_memory("step_start", &profile_label);
        let timer = Instant::now();
        let d0_key = self.signature_basis_key(s, t, subalgebra, 0);
        let d0 = self.d_matrix_signature_cached(s, t, subalgebra, 0)?;
        log_process_memory(
            "step_after_d0",
            format!(
                "{profile_label} d0_domain={} d0_target={} d0_cols={} d0_mb={:.1}",
                d0.domain_dim,
                d0.target_dim,
                d0.columns.len(),
                bytes_to_mb(d0.estimated_bytes()),
            ),
        );
        let d0_time = timer.elapsed();
        let timer = Instant::now();
        let b0 = self.d_matrix_signature_cached(s + 1, t, subalgebra, 0)?;
        log_process_memory(
            "step_after_b0",
            format!(
                "{profile_label} b0_domain={} b0_target={} b0_cols={} b0_mb={:.1}",
                b0.domain_dim,
                b0.target_dim,
                b0.columns.len(),
                bytes_to_mb(b0.estimated_bytes()),
            ),
        );
        let b0_time = timer.elapsed();
        let timer = Instant::now();
        let use_streaming_output = full_differential_output_chunk_bytes().is_some();
        if use_streaming_output {
            let d0_domain_dim = d0.domain_dim;
            let full_target_dim = if s == 0 {
                usize::from(t == 0)
            } else {
                self.basis_dimension_uncached(s - 1, t)
            };
            let output_batch_len = full_differential_output_batch_len(full_target_dim, usize::MAX);
            let mut quotient_batches = homology_representative_batches_with_label(
                &d0.columns,
                d0.target_dim,
                &b0.columns,
                d0.domain_dim,
                Some(&profile_label),
            );
            let quotient_count = quotient_batches.len();
            log_process_memory(
                "step_after_homology_batch_source",
                format!(
                    "{profile_label} quotient={} output_batch_len={output_batch_len}",
                    quotient_count,
                ),
            );
            drop(d0);
            drop(b0);
            if self.cache_policy.signature_matrix_cache == SignatureMatrixCache::DropAfterUse {
                self.signature_matrix_cache.remove(&d0_key);
            }
            release_allocator_pressure();
            log_process_memory(
                "step_after_drop_e0_matrices",
                format!("{profile_label} quotient={quotient_count}"),
            );
            let homology_time = timer.elapsed();
            if quotient_batches.is_empty() {
                if profile_detail {
                    eprintln!(
                        "ext_detail t={t} s={s} B={} added=0 d0_s={:.6} b0_s={:.6} homology_s={:.6} full_s=0.000000 lift_s=0.000000 total_s={:.6}",
                        subalgebra.name(),
                        d0_time.as_secs_f64(),
                        b0_time.as_secs_f64(),
                        homology_time.as_secs_f64(),
                        total_timer.elapsed().as_secs_f64(),
                    );
                }
                self.log_memory_cache_estimate("step_empty", s, t);
                self.clear_e0_product_cache("step_empty", s, t);
                return Ok(0);
            }

            let mut full_time = Duration::default();
            let lift_timer = Instant::now();
            let mut solver_cache = HashMap::<SignatureBasisKey, Arc<LinearSolver>>::default();
            let mut completed_lifts = Vec::<Vec<ModuleTerm>>::with_capacity(quotient_count);
            let mut initial_differential_peak_bytes = 0usize;
            let mut quotient_batch_index = 0usize;

            while let Some(quotient_batch) = quotient_batches.next_batch(output_batch_len) {
                #[cfg(test)]
                INITIAL_FULL_DIFFERENTIAL_BATCH_TEST_HITS.fetch_add(1, Ordering::Relaxed);
                let timer = Instant::now();
                let quotient_refs = quotient_batch.iter().collect::<Vec<_>>();
                let mut errors = self.apply_full_differentials_to_vectors(
                    s,
                    t,
                    subalgebra,
                    0,
                    d0_domain_dim,
                    &quotient_refs,
                )?;
                full_time += timer.elapsed();
                initial_differential_peak_bytes =
                    initial_differential_peak_bytes.max(bitvec_slice_bytes(&errors));

                let mut lifts = Vec::with_capacity(quotient_batch.len());
                for (q, error) in quotient_batch.iter().zip(errors.iter()) {
                    let x = self.signature_vector_to_terms(s, t, subalgebra, 0, q);
                    let initial_low_error =
                        self.extract_signature_vector(s.saturating_sub(1), t, subalgebra, 0, error);
                    if !initial_low_error.is_zero() {
                        let remaining = self.format_vector(s.saturating_sub(1), t, error);
                        return Err(format!(
                            "internal mismatch: E0 representative at (s,t)=({s},{t}) is not an E0-cycle for {}. Full differential: {remaining}",
                            subalgebra.name()
                        ));
                    }
                    lifts.push(x);
                }

                for sig_index in 1..subalgebra.signatures().len() {
                    let signature_timer = Instant::now();
                    let signature_error_count = errors.len();
                    let signature_output_batch_len =
                        full_differential_output_batch_len(full_target_dim, signature_error_count);
                    let mut batch_start = 0usize;
                    let mut extract_s = 0.0;
                    let mut has_error = false;
                    let mut solve_timing = SignatureSolveTiming::default();
                    let mut solve_total_s = 0.0;
                    let mut signature_full_s = 0.0;
                    let mut merge_s = 0.0;
                    let mut solved_count = 0usize;
                    let mut differential_count = 0usize;
                    let mut differential_peak_bytes = 0usize;
                    let mut domain_dim_seen = None::<usize>;

                    while batch_start < signature_error_count {
                        let batch_end =
                            (batch_start + signature_output_batch_len).min(signature_error_count);
                        let extract_timer = Instant::now();
                        let mut signature_errors = Vec::with_capacity(batch_end - batch_start);
                        let mut batch_has_error = false;
                        for error in &errors[batch_start..batch_end] {
                            let e = self.extract_signature_vector(
                                s.saturating_sub(1),
                                t,
                                subalgebra,
                                sig_index,
                                error,
                            );
                            batch_has_error |= !e.is_zero();
                            signature_errors.push(e);
                        }
                        extract_s += extract_timer.elapsed().as_secs_f64();
                        if !batch_has_error {
                            batch_start = batch_end;
                            continue;
                        }
                        has_error = true;
                        #[cfg(test)]
                        SIGNATURE_LIFT_BATCH_TEST_HITS.fetch_add(1, Ordering::Relaxed);

                        let signature_errors_mb =
                            bytes_to_mb(bitvec_slice_bytes(&signature_errors));
                        let solve_timer = Instant::now();
                        let (solutions, domain_dim, batch_solve_timing) = self
                            .solve_signature_lifts(
                                s,
                                t,
                                subalgebra,
                                sig_index,
                                &signature_errors,
                                &mut solver_cache,
                            )?;
                        solve_total_s += solve_timer.elapsed().as_secs_f64();
                        solve_timing.solver_lookup_s += batch_solve_timing.solver_lookup_s;
                        solve_timing.solve_s += batch_solve_timing.solve_s;
                        match domain_dim_seen {
                            Some(seen) if seen != domain_dim => {
                                return Err(format!(
                                    "internal error: inconsistent signature lift domain dimension at (s,t)=({s},{t}) for {} signature {}: saw {seen} and {domain_dim}",
                                    subalgebra.name(),
                                    subalgebra.signatures()[sig_index]
                                ));
                            }
                            Some(_) => {}
                            None => domain_dim_seen = Some(domain_dim),
                        }
                        if profile_verbose {
                            log_process_memory(
                                "step_after_signature_solve_quotient_batch",
                                format!(
                                    "{profile_label} sig={sig_index} quotient_batch={quotient_batch_index} batch={batch_start}-{batch_end} solutions={} signature_errors_mb={:.1}",
                                    solutions.len(),
                                    signature_errors_mb,
                                ),
                            );
                        }
                        drop(signature_errors);
                        force_release_allocator_pressure();

                        let mut solved = solutions
                            .into_iter()
                            .enumerate()
                            .filter_map(|(offset, solution)| {
                                solution.map(|solution| (batch_start + offset, solution))
                            })
                            .collect::<Vec<_>>();
                        solved_count += solved.len();
                        let timer = Instant::now();
                        let merge_timer = Instant::now();
                        for batch in solved.chunks(signature_output_batch_len) {
                            let solution_refs = batch
                                .iter()
                                .map(|(_, solution)| solution)
                                .collect::<Vec<_>>();
                            let differentials = self.apply_full_differentials_to_vectors(
                                s,
                                t,
                                subalgebra,
                                sig_index,
                                domain_dim,
                                &solution_refs,
                            )?;
                            differential_count += differentials.len();
                            differential_peak_bytes =
                                differential_peak_bytes.max(bitvec_slice_bytes(&differentials));
                            for ((index, solution), differential) in batch.iter().zip(differentials)
                            {
                                let f_terms = self.signature_vector_to_terms(
                                    s, t, subalgebra, sig_index, solution,
                                );
                                xor_terms(&mut lifts[*index], f_terms);
                                errors[*index].xor_assign(&differential);
                            }
                            force_release_allocator_pressure();
                        }
                        signature_full_s += timer.elapsed().as_secs_f64();
                        merge_s += merge_timer.elapsed().as_secs_f64();
                        solved.clear();
                        solved.shrink_to_fit();
                        force_release_allocator_pressure();
                        batch_start = batch_end;
                    }

                    if !has_error {
                        if profile_detail {
                            eprintln!(
                                "ext_signature_detail t={t} s={s} B={} sig={sig_index} signature={} errors={} nonzero=false extract_s={extract_s:.6} solver_lookup_s=0.000000 solve_s=0.000000 full_s=0.000000 merge_s=0.000000 total_s={:.6}",
                                subalgebra.name(),
                                subalgebra.signatures()[sig_index],
                                signature_error_count,
                                signature_timer.elapsed().as_secs_f64(),
                            );
                        }
                        continue;
                    }

                    full_time += Duration::from_secs_f64(signature_full_s);
                    if profile_verbose {
                        log_process_memory(
                            "step_after_signature_full_differential_quotient_batch",
                            format!(
                                "{profile_label} sig={sig_index} quotient_batch={quotient_batch_index} solution_refs={} differentials={} differentials_peak_mb={:.1} output_batch_len={signature_output_batch_len}",
                                solved_count,
                                differential_count,
                                bytes_to_mb(differential_peak_bytes),
                            ),
                        );
                    }
                    if profile_detail {
                        eprintln!(
                            "ext_signature_detail t={t} s={s} B={} sig={sig_index} signature={} errors={} nonzero=true extract_s={extract_s:.6} solver_lookup_s={:.6} solve_s={:.6} solve_total_s={solve_total_s:.6} full_s={signature_full_s:.6} merge_s={merge_s:.6} total_s={:.6}",
                            subalgebra.name(),
                            subalgebra.signatures()[sig_index],
                            signature_error_count,
                            solve_timing.solver_lookup_s,
                            solve_timing.solve_s,
                            signature_timer.elapsed().as_secs_f64(),
                        );
                    }
                }

                for (x, error) in lifts.into_iter().zip(errors) {
                    if !error.is_zero() {
                        let remaining = self.format_vector(s.saturating_sub(1), t, &error);
                        return Err(format!(
                            "Algorithm 2 produced a non-cycle at (s,t)=({s},{t}) for {}. Remaining error: {remaining}. Use --algorithm naive to compare, or use a bidegree where the Algorithm 2 lower-line condition applies.",
                            subalgebra.name(),
                        ));
                    }
                    completed_lifts.push(x);
                }
                force_release_allocator_pressure();
                quotient_batch_index += 1;
            }
            drop(quotient_batches);
            force_release_allocator_pressure();

            let added = completed_lifts.len();
            for x in completed_lifts {
                self.add_generator(s + 1, t, x);
            }
            let lift_time = lift_timer.elapsed();
            log_process_memory(
                "step_after_quotient_batch_lift",
                format!(
                    "{profile_label} quotient={quotient_count} added={added} output_batch_len={output_batch_len} initial_errors_peak_mb={:.1}",
                    bytes_to_mb(initial_differential_peak_bytes),
                ),
            );
            if profile_detail {
                eprintln!(
                    "ext_detail t={t} s={s} B={} added={added} d0_s={:.6} b0_s={:.6} homology_s={:.6} full_s={:.6} lift_s={:.6} total_s={:.6}",
                    subalgebra.name(),
                    d0_time.as_secs_f64(),
                    b0_time.as_secs_f64(),
                    homology_time.as_secs_f64(),
                    full_time.as_secs_f64(),
                    lift_time.as_secs_f64(),
                    total_timer.elapsed().as_secs_f64(),
                );
            }
            self.log_memory_cache_estimate("step_end", s, t);
            self.clear_e0_product_cache("step_end", s, t);
            return Ok(added);
        }

        let quotient = homology_representatives_with_label(
            &d0.columns,
            d0.target_dim,
            &b0.columns,
            d0.domain_dim,
            Some(&profile_label),
        );
        let d0_domain_dim = d0.domain_dim;
        log_process_memory(
            "step_after_homology",
            format!(
                "{profile_label} quotient={} quotient_mb={:.1}",
                quotient.len(),
                bytes_to_mb(bitvec_slice_bytes(&quotient)),
            ),
        );
        drop(d0);
        drop(b0);
        if self.cache_policy.signature_matrix_cache == SignatureMatrixCache::DropAfterUse {
            self.signature_matrix_cache.remove(&d0_key);
        }
        release_allocator_pressure();
        log_process_memory(
            "step_after_drop_e0_matrices",
            format!("{profile_label} quotient={}", quotient.len()),
        );
        let homology_time = timer.elapsed();
        if quotient.is_empty() {
            if profile_detail {
                eprintln!(
                    "ext_detail t={t} s={s} B={} added=0 d0_s={:.6} b0_s={:.6} homology_s={:.6} full_s=0.000000 lift_s=0.000000 total_s={:.6}",
                    subalgebra.name(),
                    d0_time.as_secs_f64(),
                    b0_time.as_secs_f64(),
                    homology_time.as_secs_f64(),
                    total_timer.elapsed().as_secs_f64(),
                );
            }
            self.log_memory_cache_estimate("step_empty", s, t);
            self.clear_e0_product_cache("step_empty", s, t);
            return Ok(0);
        }

        let mut lifts = Vec::with_capacity(quotient.len());

        let timer = Instant::now();
        let quotient_refs = quotient.iter().collect::<Vec<_>>();
        let mut errors = self.apply_full_differentials_to_vectors(
            s,
            t,
            subalgebra,
            0,
            d0_domain_dim,
            &quotient_refs,
        )?;
        log_process_memory(
            "step_after_initial_full_differential",
            format!(
                "{profile_label} quotient={} errors={} errors_mb={:.1}",
                quotient_refs.len(),
                errors.len(),
                bytes_to_mb(bitvec_slice_bytes(&errors)),
            ),
        );
        for (q, error) in quotient.iter().zip(errors.iter()) {
            let x = self.signature_vector_to_terms(s, t, subalgebra, 0, q);
            let initial_low_error =
                self.extract_signature_vector(s.saturating_sub(1), t, subalgebra, 0, error);
            if !initial_low_error.is_zero() {
                let remaining = self.format_vector(s.saturating_sub(1), t, error);
                return Err(format!(
                    "internal mismatch: E0 representative at (s,t)=({s},{t}) is not an E0-cycle for {}. Full differential: {remaining}",
                    subalgebra.name()
                ));
            }

            lifts.push(x);
        }
        let mut full_time = timer.elapsed();
        drop(quotient);

        let lift_timer = Instant::now();
        let mut solver_cache = HashMap::<SignatureBasisKey, Arc<LinearSolver>>::default();
        for sig_index in 1..subalgebra.signatures().len() {
            let signature_timer = Instant::now();

            let extract_timer = Instant::now();
            let mut signature_errors = Vec::with_capacity(errors.len());
            let mut has_error = false;
            for error in &errors {
                let e = self.extract_signature_vector(
                    s.saturating_sub(1),
                    t,
                    subalgebra,
                    sig_index,
                    error,
                );
                has_error |= !e.is_zero();
                signature_errors.push(e);
            }
            let extract_s = extract_timer.elapsed().as_secs_f64();
            if !has_error {
                if profile_detail {
                    eprintln!(
                        "ext_signature_detail t={t} s={s} B={} sig={sig_index} signature={} errors={} nonzero=false extract_s={extract_s:.6} solver_lookup_s=0.000000 solve_s=0.000000 full_s=0.000000 merge_s=0.000000 total_s={:.6}",
                        subalgebra.name(),
                        subalgebra.signatures()[sig_index],
                        signature_errors.len(),
                        signature_timer.elapsed().as_secs_f64(),
                    );
                }
                continue;
            }

            let signature_error_count = signature_errors.len();
            let signature_errors_mb = bytes_to_mb(bitvec_slice_bytes(&signature_errors));
            let solve_timer = Instant::now();
            let (solutions, domain_dim, solve_timing) = self.solve_signature_lifts(
                s,
                t,
                subalgebra,
                sig_index,
                &signature_errors,
                &mut solver_cache,
            )?;
            let solve_total_s = solve_timer.elapsed().as_secs_f64();
            if profile_verbose {
                log_process_memory(
                    "step_after_signature_solve",
                    format!(
                        "{profile_label} sig={sig_index} solutions={} signature_errors_mb={:.1}",
                        solutions.len(),
                        signature_errors_mb,
                    ),
                );
            }
            drop(signature_errors);
            release_allocator_pressure();

            let mut solved = solutions
                .into_iter()
                .enumerate()
                .filter_map(|(index, solution)| solution.map(|solution| (index, solution)))
                .collect::<Vec<_>>();
            let full_target_dim = if s == 0 {
                usize::from(t == 0)
            } else {
                self.basis_dimension_uncached(s - 1, t)
            };
            let output_batch_len =
                full_differential_output_batch_len(full_target_dim, solved.len());
            let timer = Instant::now();
            let merge_timer = Instant::now();
            let mut differential_count = 0usize;
            let mut differential_peak_bytes = 0usize;
            for batch in solved.chunks(output_batch_len) {
                let solution_refs = batch
                    .iter()
                    .map(|(_, solution)| solution)
                    .collect::<Vec<_>>();
                let differentials = self.apply_full_differentials_to_vectors(
                    s,
                    t,
                    subalgebra,
                    sig_index,
                    domain_dim,
                    &solution_refs,
                )?;
                differential_count += differentials.len();
                differential_peak_bytes =
                    differential_peak_bytes.max(bitvec_slice_bytes(&differentials));
                for ((index, solution), differential) in batch.iter().zip(differentials) {
                    let f_terms =
                        self.signature_vector_to_terms(s, t, subalgebra, sig_index, solution);
                    xor_terms(&mut lifts[*index], f_terms);
                    errors[*index].xor_assign(&differential);
                }
            }
            solved.clear();
            solved.shrink_to_fit();
            let signature_full_s = timer.elapsed().as_secs_f64();
            full_time += Duration::from_secs_f64(signature_full_s);
            let merge_s = merge_timer.elapsed().as_secs_f64();
            if profile_verbose {
                log_process_memory(
                    "step_after_signature_full_differential",
                    format!(
                        "{profile_label} sig={sig_index} solution_refs={} differentials={} differentials_peak_mb={:.1} output_batch_len={output_batch_len}",
                        differential_count,
                        differential_count,
                        bytes_to_mb(differential_peak_bytes),
                    ),
                );
            }
            if profile_detail {
                eprintln!(
                    "ext_signature_detail t={t} s={s} B={} sig={sig_index} signature={} errors={} nonzero=true extract_s={extract_s:.6} solver_lookup_s={:.6} solve_s={:.6} solve_total_s={solve_total_s:.6} full_s={signature_full_s:.6} merge_s={merge_s:.6} total_s={:.6}",
                    subalgebra.name(),
                    subalgebra.signatures()[sig_index],
                    signature_error_count,
                    solve_timing.solver_lookup_s,
                    solve_timing.solve_s,
                    signature_timer.elapsed().as_secs_f64(),
                );
            }
        }
        let lift_time = lift_timer.elapsed();

        let mut added = 0;
        for (x, error) in lifts.into_iter().zip(errors) {
            if !error.is_zero() {
                let remaining = self.format_vector(s.saturating_sub(1), t, &error);
                return Err(format!(
                    "Algorithm 2 produced a non-cycle at (s,t)=({s},{t}) for {}. Remaining error: {remaining}. Use --algorithm naive to compare, or use a bidegree where the Algorithm 2 lower-line condition applies.",
                    subalgebra.name(),
                ));
            }
            self.add_generator(s + 1, t, x);
            added += 1;
        }
        if profile_detail {
            eprintln!(
                "ext_detail t={t} s={s} B={} added={added} d0_s={:.6} b0_s={:.6} homology_s={:.6} full_s={:.6} lift_s={:.6} total_s={:.6}",
                subalgebra.name(),
                d0_time.as_secs_f64(),
                b0_time.as_secs_f64(),
                homology_time.as_secs_f64(),
                full_time.as_secs_f64(),
                lift_time.as_secs_f64(),
                total_timer.elapsed().as_secs_f64(),
            );
        }
        self.log_memory_cache_estimate("step_end", s, t);
        self.clear_e0_product_cache("step_end", s, t);
        Ok(added)
    }

    fn solve_signature_lifts(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
        signature_errors: &[BitVec],
        solver_cache: &mut HashMap<SignatureBasisKey, Arc<LinearSolver>>,
    ) -> Result<(Vec<Option<BitVec>>, usize, SignatureSolveTiming), String> {
        if subalgebra.profile_cache_key().is_some() && s > 0 {
            let sig_degree = subalgebra.signatures()[sig_index].degree();
            if let Some(shifted_t) = t.checked_sub(sig_degree) {
                let source_translation =
                    self.signature_to_zero_translation(s, t, subalgebra, sig_index)?;
                let target_translation =
                    self.signature_to_zero_translation(s - 1, t, subalgebra, sig_index)?;
                let solver_timer = Instant::now();
                let solver =
                    self.linear_solver_cached(s, shifted_t, subalgebra, 0, solver_cache)?;
                let solver_lookup_s = solver_timer.elapsed().as_secs_f64();
                if solver.domain_dim() != source_translation.zero_len
                    || solver.target_dim() != target_translation.zero_len
                {
                    return Err(format!(
                        "internal error: signature translation dimension mismatch at (s,t)=({s},{t}) for {} signature {}",
                        subalgebra.name(),
                        subalgebra.signatures()[sig_index]
                    ));
                }
                let solve_timer = Instant::now();
                let solutions = signature_errors
                    .par_iter()
                    .map(|e| {
                        if e.is_zero() {
                            return Ok(None);
                        }
                        let shifted_error;
                        let shifted_error_ref = if target_translation.identity {
                            e
                        } else {
                            shifted_error = translate_to_zero(e, &target_translation);
                            &shifted_error
                        };
                        let Some(shifted_f) = solver.solve(shifted_error_ref) else {
                            return Err(format!(
                                "Algorithm 2 failed to solve a lifting problem at (s,t)=({s},{t}) for {} signature {}. The subalgebra may be outside the valid range.",
                                subalgebra.name(),
                                subalgebra.signatures()[sig_index]
                            ));
                        };
                        let f = if source_translation.identity {
                            shifted_f
                        } else {
                            translate_from_zero(&shifted_f, &source_translation)
                        };
                        Ok(Some(f))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                return Ok((
                    solutions,
                    source_translation.current_len,
                    SignatureSolveTiming {
                        solver_lookup_s,
                        solve_s: solve_timer.elapsed().as_secs_f64(),
                    },
                ));
            }
        }

        let solver_timer = Instant::now();
        let solver = self.linear_solver_cached(s, t, subalgebra, sig_index, solver_cache)?;
        let solver_lookup_s = solver_timer.elapsed().as_secs_f64();
        let domain_dim = solver.domain_dim();
        let solve_timer = Instant::now();
        let solutions = signature_errors
            .par_iter()
            .map(|e| {
                if e.is_zero() {
                    return Ok(None);
                }
                let Some(f) = solver.solve(e) else {
                    return Err(format!(
                        "Algorithm 2 failed to solve a lifting problem at (s,t)=({s},{t}) for {} signature {}. The subalgebra may be outside the valid range.",
                        subalgebra.name(),
                        subalgebra.signatures()[sig_index]
                    ));
                };
                Ok(Some(f))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok((
            solutions,
            domain_dim,
            SignatureSolveTiming {
                solver_lookup_s,
                solve_s: solve_timer.elapsed().as_secs_f64(),
            },
        ))
    }

    fn linear_solver_cached(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
        solver_cache: &mut HashMap<SignatureBasisKey, Arc<LinearSolver>>,
    ) -> Result<Arc<LinearSolver>, String> {
        let key = self.signature_basis_key(s, t, subalgebra, sig_index);
        if let Some(cached) = solver_cache.get(&key) {
            return Ok(Arc::clone(cached));
        }
        if let Some(cached) = self.signature_solver_cache.get(&key) {
            let solver = Arc::clone(cached);
            solver_cache.insert(key, Arc::clone(&solver));
            return Ok(solver);
        }
        let build_timer = Instant::now();
        let matrix_timer = Instant::now();
        let matrix = self.d_matrix_signature_cached(s, t, subalgebra, sig_index)?;
        let matrix_s = matrix_timer.elapsed().as_secs_f64();
        let solver_timer = Instant::now();
        let solver = Arc::new(LinearSolver::new(&matrix.columns, matrix.target_dim));
        let solver_s = solver_timer.elapsed().as_secs_f64();
        if std::env::var_os("EXT_PROFILE_SIGNATURE_DETAIL").is_some() {
            eprintln!(
                "ext_solver_build t={t} s={s} B={} sig={sig_index} signature={} domain={} target={} pivots={} matrix_s={matrix_s:.6} solver_s={solver_s:.6} total_s={:.6}",
                subalgebra.name(),
                subalgebra.signatures()[sig_index],
                matrix.domain_dim,
                matrix.target_dim,
                solver.pivot_count(),
                build_timer.elapsed().as_secs_f64(),
            );
        }
        drop(matrix);
        if self.cache_policy.signature_matrix_cache == SignatureMatrixCache::DropAfterUse {
            self.signature_matrix_cache.remove(&key);
        }
        self.signature_solver_cache.insert(key, Arc::clone(&solver));
        solver_cache.insert(key, Arc::clone(&solver));
        Ok(solver)
    }

    fn naive_homology_representatives(
        &mut self,
        s: usize,
        t: usize,
    ) -> Result<Vec<BitVec>, String> {
        let d = self.d_matrix(s, t)?;
        let boundaries = self.d_matrix(s + 1, t)?;
        Ok(homology_representatives(
            &d.columns,
            d.target_dim,
            &boundaries.columns,
            d.domain_dim,
        ))
    }

    fn d_matrix(&mut self, s: usize, t: usize) -> Result<LinearMap, String> {
        let domain = self.basis_cached(s, t);
        if s == 0 {
            let target_dim = usize::from(t == 0);
            let mut columns = Vec::with_capacity(domain.len());
            for elem in domain.iter() {
                let mut col = BitVec::new(target_dim);
                if t == 0 && elem.generator == 0 && elem.coeff_packed == 0 {
                    col.set(0, true);
                }
                columns.push(col);
            }
            return Ok(LinearMap {
                columns,
                domain_dim: domain.len(),
                target_dim,
            });
        }

        let target = self.basis_cached(s - 1, t);
        let target_index = self.basis_index_cached(s - 1, t);
        let mut columns = Vec::with_capacity(domain.len());

        for elem in domain.iter() {
            let col =
                self.differential_of_basis_elem_packed(elem, target_index.as_ref(), target.len())?;
            columns.push(col);
        }

        Ok(LinearMap {
            columns,
            domain_dim: domain.len(),
            target_dim: target.len(),
        })
    }

    fn d_matrix_signature(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
    ) -> Result<LinearMap, String> {
        let profile_detail = std::env::var_os("EXT_PROFILE_SIGNATURE_DETAIL").is_some();
        let matrix_timer = Instant::now();
        let domain_timer = Instant::now();
        let domain = self.basis_signature_cached(s, t, subalgebra, sig_index);
        let domain_s = domain_timer.elapsed().as_secs_f64();
        if s == 0 {
            let target_dim = usize::from(t == 0 && sig_index == 0);
            let columns_timer = Instant::now();
            let mut columns = Vec::with_capacity(domain.len());
            for elem in domain.iter() {
                let mut col = BitVec::new(target_dim);
                if t == 0 && elem.generator == 0 && elem.coeff_packed == 0 && sig_index == 0 {
                    col.set(0, true);
                }
                columns.push(col);
            }
            if profile_detail {
                eprintln!(
                    "ext_matrix_detail t={t} s={s} B={} sig={sig_index} signature={} domain={} target={} domain_s={domain_s:.6} target_s=0.000000 index_s=0.000000 product_s=0.000000 columns_s={:.6} total_s={:.6}",
                    subalgebra.name(),
                    subalgebra.signatures()[sig_index],
                    domain.len(),
                    target_dim,
                    columns_timer.elapsed().as_secs_f64(),
                    matrix_timer.elapsed().as_secs_f64(),
                );
            }
            return Ok(LinearMap {
                columns,
                domain_dim: domain.len(),
                target_dim,
            });
        }

        let target_timer = Instant::now();
        let target = self.basis_signature_cached(s - 1, t, subalgebra, sig_index);
        let target_s = target_timer.elapsed().as_secs_f64();
        let index_timer = Instant::now();
        let target_index = self.basis_signature_index_cached(s - 1, t, subalgebra, sig_index);
        let index_s = index_timer.elapsed().as_secs_f64();
        let product_timer = Instant::now();
        self.precompute_products_for_domain(domain.as_slice(), Some(subalgebra), s, t, sig_index);
        let product_s = product_timer.elapsed().as_secs_f64();
        let profile_key = subalgebra.profile_cache_key();
        let profile_signature = profile_key.map(|_| {
            subalgebra.signatures()[sig_index]
                .packed()
                .expect("profile signature cannot be packed")
        });
        let columns_timer = Instant::now();
        let columns = domain
            .par_iter()
            .map(|elem| {
                debug_assert_eq!(
                    subalgebra.signature_index_packed(elem.coeff_packed),
                    sig_index
                );
                let raw_left_key = elem.coeff_packed;
                let left_key = profile_key
                    .map(|_| subalgebra.profile_quotient_packed_unchecked(raw_left_key))
                    .unwrap_or(raw_left_key);
                let generator = &self.generators[elem.generator];
                let mut col = BitVec::new(target.len());
                for term in generator.differential.iter() {
                    let right_key = term.coeff_packed;
                    if let Some(signature) = profile_signature {
                        let key = (left_key, right_key);
                        if let Some(products) = self.mult_e0_cache.get(&key) {
                            let mut missing_coeff = None;
                            for &product in products.as_slice() {
                                let Some(coeff_packed) = subalgebra
                                    .attach_profile_signature_packed_unchecked(signature, product)
                                else {
                                    continue;
                                };
                                let image_key = (coeff_packed, term.generator);
                                if let Some(&row) = target_index.get(&image_key) {
                                    col.toggle(row);
                                } else {
                                    missing_coeff.get_or_insert(coeff_packed);
                                }
                            }
                            if let Some(coeff_packed) = missing_coeff {
                                let coeff = Milnor::from_packed(coeff_packed);
                                return Err(format!(
                                    "internal error: signature image term {}*g{} missing from E_{} C_{},{}",
                                    coeff,
                                    term.generator,
                                    sig_index,
                                    s - 1,
                                    t
                                ));
                            }
                        }
                    } else {
                        let key = (left_key, right_key);
                        let products = self
                            .mult_packed_cache
                            .get(&key)
                            .expect("matrix product was precomputed");
                        for &coeff_packed in products {
                            let product_sig = subalgebra.signature_index_packed(coeff_packed);
                            if product_sig == sig_index {
                                let image_key = (coeff_packed, term.generator);
                                let Some(&row) = target_index.get(&image_key) else {
                                    let coeff = Milnor::from_packed(coeff_packed);
                                    return Err(format!(
                                        "internal error: signature image term {}*g{} missing from E_{} C_{},{}",
                                        coeff,
                                        term.generator,
                                        sig_index,
                                        s - 1,
                                        t
                                    ));
                                };
                                col.toggle(row);
                            } else if product_sig < sig_index {
                                let coeff = Milnor::from_packed(coeff_packed);
                                return Err(format!(
                                    "signature order violation for {}: {} moved from signature {} to {}",
                                    subalgebra.name(),
                                    coeff,
                                    sig_index,
                                    product_sig
                                ));
                            }
                        }
                    }
                }
                Ok(col)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let columns_s = columns_timer.elapsed().as_secs_f64();
        if profile_detail {
            eprintln!(
                "ext_matrix_detail t={t} s={s} B={} sig={sig_index} signature={} domain={} target={} domain_s={domain_s:.6} target_s={target_s:.6} index_s={index_s:.6} product_s={product_s:.6} columns_s={columns_s:.6} total_s={:.6}",
                subalgebra.name(),
                subalgebra.signatures()[sig_index],
                domain.len(),
                target.len(),
                matrix_timer.elapsed().as_secs_f64(),
            );
        }

        Ok(LinearMap {
            columns,
            domain_dim: domain.len(),
            target_dim: target.len(),
        })
    }

    fn d_matrix_signature_cached(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
    ) -> Result<Arc<LinearMap>, String> {
        let key = self.signature_basis_key(s, t, subalgebra, sig_index);
        if let Some(cached) = self.signature_matrix_cache.get(&key) {
            return Ok(Arc::clone(cached));
        }
        let matrix = Arc::new(self.d_matrix_signature(s, t, subalgebra, sig_index)?);
        self.signature_matrix_cache.insert(key, Arc::clone(&matrix));
        Ok(matrix)
    }

    fn apply_full_differentials_to_vectors(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
        domain_dim: usize,
        vectors: &[&BitVec],
    ) -> Result<Vec<BitVec>, String> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let profile_full_timing = std::env::var_os("EXT_PROFILE_FULL_TIMING").is_some();
        let total_timer = profile_full_timing.then(Instant::now);

        let target_timer = profile_full_timing.then(Instant::now);
        let target_lookup = (s > 0).then(|| self.full_basis_lookup_cached(s - 1, t));
        let target_dim = target_lookup
            .as_ref()
            .map(|lookup| lookup.target_dim)
            .unwrap_or_else(|| usize::from(t == 0));
        let target_s = target_timer
            .map(|timer| timer.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let uses_timer = profile_full_timing.then(Instant::now);
        let mut uses = Vec::<(usize, usize)>::new();
        for (vector_index, vector) in vectors.iter().enumerate() {
            debug_assert_eq!(vector.len(), domain_dim);
            uses.extend(vector.ones().map(|source| (source, vector_index)));
        }
        let uses_s = uses_timer
            .map(|timer| timer.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let mut outputs = (0..vectors.len())
            .map(|_| BitVec::new(target_dim))
            .collect::<Vec<_>>();

        if uses.is_empty() {
            if let Some(total_timer) = total_timer {
                eprintln!(
                    "ext_full_timing t={t} s={s} B={} sig={sig_index} vectors={} uses=0 groups=0 target_dim={} uses_s={uses_s:.6} target_s={target_s:.6} sort_s=0.000000 group_s=0.000000 domain_s=0.000000 columns_s=0.000000 merge_s=0.000000 total_s={:.6}",
                    subalgebra.name(),
                    vectors.len(),
                    target_dim,
                    total_timer.elapsed().as_secs_f64(),
                );
            }
            return Ok(outputs);
        }

        let sort_timer = profile_full_timing.then(Instant::now);
        uses.sort_unstable();
        let sort_s = sort_timer
            .map(|timer| timer.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let group_timer = profile_full_timing.then(Instant::now);
        let mut groups = Vec::<(usize, usize, usize)>::new();
        let mut start = 0;
        while start < uses.len() {
            let source = uses[start].0;
            let mut end = start + 1;
            while end < uses.len() && uses[end].0 == source {
                end += 1;
            }
            groups.push((source, start, end));
            start = end;
        }
        let group_s = group_timer
            .map(|timer| timer.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        let domain_timer = profile_full_timing.then(Instant::now);
        let domain = self.basis_signature_cached(s, t, subalgebra, sig_index);
        let domain_s = domain_timer
            .map(|timer| timer.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        if domain.len() != domain_dim {
            return Err(format!(
                "internal error: signature domain dimension mismatch at (s,t)=({s},{t}) for {} signature {}: got {}, expected {}",
                subalgebra.name(),
                subalgebra.signatures()[sig_index],
                domain_dim,
                domain.len()
            ));
        }
        match groups.last() {
            Some(&(source, _, _)) if source >= domain.len() => {
                return Err(format!(
                    "internal error: selected source column {source} outside signature domain of length {}",
                    domain.len()
                ));
            }
            _ => {}
        }

        let bytes_per_column = BitVec::storage_bytes_for_len(target_dim);
        let full_differential_chunk_bytes = full_differential_chunk_bytes();
        let columns_per_chunk = full_differential_chunk_bytes
            .checked_div(bytes_per_column)
            .unwrap_or_else(|| groups.len().max(1))
            .max(1);
        let chunk_count = groups.len().div_ceil(columns_per_chunk);
        if std::env::var_os("EXT_PROFILE_MEMORY").is_some() {
            let old_columns_mb = groups.len() as f64 * bytes_per_column as f64 / 1_048_576.0;
            let chunk_mb =
                columns_per_chunk.min(groups.len()) as f64 * bytes_per_column as f64 / 1_048_576.0;
            let outputs_mb = vectors.len() as f64 * bytes_per_column as f64 / 1_048_576.0;
            let uses_mb =
                uses.capacity() as f64 * std::mem::size_of::<(usize, usize)>() as f64 / 1_048_576.0;
            let verbose = std::env::var_os("EXT_PROFILE_MEMORY_VERBOSE").is_some();
            let large = old_columns_mb >= 128.0 || chunk_mb >= 128.0 || outputs_mb >= 128.0;
            if verbose || large {
                eprintln!(
                    "ext_mem_full t={t} s={s} sig={sig_index} vectors={} uses={} selected={} target_dim={} bytes_per_column={} chunk_columns={} old_columns_mb={old_columns_mb:.1} chunk_mb={chunk_mb:.1} outputs_mb={outputs_mb:.1} uses_mb={uses_mb:.1}",
                    vectors.len(),
                    uses.len(),
                    groups.len(),
                    target_dim,
                    bytes_per_column,
                    columns_per_chunk,
                );
            }
        }

        let mut columns_s = 0.0;
        let mut merge_s = 0.0;
        if s == 0 {
            for chunk in groups.chunks(columns_per_chunk) {
                let columns_timer = profile_full_timing.then(Instant::now);
                let columns = chunk
                    .iter()
                    .map(|&(source, _, _)| {
                        let elem = &domain[source];
                        let mut col = BitVec::new(target_dim);
                        if t == 0 && elem.generator == 0 && elem.coeff_packed == 0 {
                            col.set(0, true);
                        }
                        col
                    })
                    .collect::<Vec<_>>();
                if let Some(timer) = columns_timer {
                    columns_s += timer.elapsed().as_secs_f64();
                }
                let merge_timer = profile_full_timing.then(Instant::now);
                xor_full_columns_into_outputs(&columns, chunk, &uses, &mut outputs);
                if let Some(timer) = merge_timer {
                    merge_s += timer.elapsed().as_secs_f64();
                }
            }
            if let Some(total_timer) = total_timer {
                eprintln!(
                    "ext_full_timing t={t} s={s} B={} sig={sig_index} vectors={} uses={} groups={} target_dim={} bytes_per_column={} chunk_mb={} chunk_columns={} chunks={} rayon_threads={} uses_s={uses_s:.6} target_s={target_s:.6} sort_s={sort_s:.6} group_s={group_s:.6} domain_s={domain_s:.6} columns_s={columns_s:.6} merge_s={merge_s:.6} total_s={:.6}",
                    subalgebra.name(),
                    vectors.len(),
                    uses.len(),
                    groups.len(),
                    target_dim,
                    bytes_per_column,
                    full_differential_chunk_bytes / (1024 * 1024),
                    columns_per_chunk,
                    chunk_count,
                    rayon::current_num_threads(),
                    total_timer.elapsed().as_secs_f64(),
                );
            }
            return Ok(outputs);
        }

        let target_lookup =
            target_lookup.expect("positive homological degree has a full target lookup");
        let generators = &self.generators;
        let mut plan_cache = vec![None; generators.len()];
        for &(source, _, _) in &groups {
            let elem = &domain[source];
            if plan_cache[elem.generator].is_none() {
                plan_cache[elem.generator] = Some(Arc::new(Self::full_differential_term_plan(
                    t,
                    elem.generator,
                    generators,
                    target_lookup.as_ref(),
                )?));
            }
        }
        for chunk in groups.chunks(columns_per_chunk) {
            let columns_timer = profile_full_timing.then(Instant::now);
            let columns = chunk
                .par_iter()
                .map(|&(source, _, _)| {
                    let elem = &domain[source];
                    let plans = plan_cache[elem.generator]
                        .as_ref()
                        .expect("full differential plan was precomputed");
                    Self::full_differential_column_from_plan(
                        elem.coeff_packed,
                        plans.as_slice(),
                        target_lookup.target_dim,
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            if let Some(timer) = columns_timer {
                columns_s += timer.elapsed().as_secs_f64();
            }
            let merge_timer = profile_full_timing.then(Instant::now);
            xor_full_columns_into_outputs(&columns, chunk, &uses, &mut outputs);
            if let Some(timer) = merge_timer {
                merge_s += timer.elapsed().as_secs_f64();
            }
        }

        if let Some(total_timer) = total_timer {
            eprintln!(
                "ext_full_timing t={t} s={s} B={} sig={sig_index} vectors={} uses={} groups={} target_dim={} bytes_per_column={} chunk_mb={} chunk_columns={} chunks={} rayon_threads={} uses_s={uses_s:.6} target_s={target_s:.6} sort_s={sort_s:.6} group_s={group_s:.6} domain_s={domain_s:.6} columns_s={columns_s:.6} merge_s={merge_s:.6} total_s={:.6}",
                subalgebra.name(),
                vectors.len(),
                uses.len(),
                groups.len(),
                target_dim,
                bytes_per_column,
                full_differential_chunk_bytes / (1024 * 1024),
                columns_per_chunk,
                chunk_count,
                rayon::current_num_threads(),
                total_timer.elapsed().as_secs_f64(),
            );
        }

        Ok(outputs)
    }

    fn full_differential_term_plan(
        t: usize,
        generator_id: usize,
        generators: &[Generator],
        target_lookup: &FullBasisLookup,
    ) -> Result<Vec<FullDifferentialTermPlan>, String> {
        let generator = &generators[generator_id];
        let mut plans = Vec::with_capacity(generator.differential.len());
        for term in generator.differential.iter() {
            let Some(rows) = target_lookup
                .generator_rows
                .get(term.generator)
                .and_then(Option::as_ref)
            else {
                return Err(format!(
                    "internal error: full image generator g{} missing",
                    term.generator
                ));
            };
            let product_degree_bound =
                t.checked_sub(generators[term.generator].t).ok_or_else(|| {
                    format!(
                        "internal error: full image generator g{} has internal degree above t={t}",
                        term.generator
                    )
                })?;
            plans.push(FullDifferentialTermPlan {
                coeff_packed: term.coeff_packed,
                target_generator: term.generator,
                rows_offset: rows.offset,
                coeff_index: Arc::clone(&rows.coeff_index),
                product_width_bound: bounded_product_width(product_degree_bound),
                right_width: packed_support_width(term.coeff_packed),
            });
        }
        Ok(plans)
    }

    fn full_differential_column_from_plan(
        elem_coeff_packed: CoeffKey,
        plans: &[FullDifferentialTermPlan],
        target_dim: usize,
    ) -> Result<BitVec, String> {
        let mut col = BitVec::new(target_dim);
        let source_width = packed_support_width(elem_coeff_packed);
        for plan in plans {
            let mut missing_coeff = None;
            multiply_packed_fast_raw_for_each_width(
                elem_coeff_packed,
                plan.coeff_packed,
                plan.product_width_bound
                    .min(source_width.saturating_add(plan.right_width)),
                |coeff_packed| {
                    if let Some(&local) = plan.coeff_index.get(&coeff_packed) {
                        col.toggle(plan.rows_offset + local);
                    } else {
                        missing_coeff.get_or_insert(coeff_packed);
                    }
                },
            );
            if let Some(coeff_packed) = missing_coeff {
                let coeff = Milnor::from_packed(coeff_packed);
                return Err(format!(
                    "internal error: full image term {coeff}*g{} missing",
                    plan.target_generator
                ));
            }
        }
        Ok(col)
    }

    fn differential_vector_from_terms(
        &mut self,
        s: usize,
        t: usize,
        terms: &[ModuleTerm],
    ) -> Result<BitVec, String> {
        if s == 0 {
            let mut out = BitVec::new(usize::from(t == 0));
            if t == 0 {
                for term in terms {
                    if term.generator == 0 && term.coeff_packed == 0 {
                        out.toggle(0);
                    }
                }
            }
            return Ok(out);
        }

        let target = self.basis_cached(s - 1, t);
        let target_index = self.basis_index_cached(s - 1, t);
        let mut out = BitVec::new(target.len());
        for term in terms {
            let elem = BasisElem {
                coeff_packed: term.coeff_packed,
                generator: term.generator,
            };
            let col =
                self.differential_of_basis_elem_packed(&elem, target_index.as_ref(), target.len())?;
            out.xor_assign(&col);
        }
        Ok(out)
    }

    fn differential_of_basis_elem_packed(
        &mut self,
        elem: &BasisElem,
        target_index: &HashMap<(CoeffKey, usize), usize>,
        target_dim: usize,
    ) -> Result<BitVec, String> {
        let generator = &self.generators[elem.generator];
        let diff = generator.differential.clone();
        let mut col = BitVec::new(target_dim);
        for term in diff.iter() {
            let products = self
                .cached_multiply_packed_keys(elem.coeff_packed, term.coeff_packed)
                .to_vec();
            for coeff_packed in products {
                let image_key = (coeff_packed, term.generator);
                let Some(&row) = target_index.get(&image_key) else {
                    let coeff = Milnor::from_packed(coeff_packed);
                    return Err(format!(
                        "internal error: image term {}*{} missing",
                        coeff,
                        self.generator_name(term.generator)
                    ));
                };
                col.toggle(row);
            }
        }
        Ok(col)
    }

    fn extract_signature_vector(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
        full_vector: &BitVec,
    ) -> BitVec {
        let sig_basis_len = self
            .basis_signature_cached(s, t, subalgebra, sig_index)
            .len();
        let routing = self.basis_signature_routing_cached(s, t, subalgebra);
        let mut out = BitVec::new(sig_basis_len);
        for i in full_vector.ones() {
            let (sig, local) = routing[i];
            if sig == sig_index {
                out.toggle(local);
            }
        }
        out
    }

    fn basis(&self, s: usize, t: usize) -> Vec<BasisElem> {
        self.build_basis(s, t)
    }

    fn build_basis(&self, s: usize, t: usize) -> Vec<BasisElem> {
        let mut out = Vec::new();
        let Some(gens) = self.gens_by_s.get(s) else {
            return out;
        };
        for &generator_id in gens {
            let generator = &self.generators[generator_id];
            if generator.t > t {
                continue;
            }
            let alg_degree = t - generator.t;
            for &coeff_packed in &self.algebra_basis[alg_degree] {
                out.push(BasisElem {
                    coeff_packed,
                    generator: generator_id,
                });
            }
        }
        out
    }

    fn basis_cached(&mut self, s: usize, t: usize) -> Arc<Vec<BasisElem>> {
        let key = (s, t);
        if let Some(cached) = self.basis_cache.get(&key) {
            return Arc::clone(cached);
        }
        let basis = Arc::new(self.build_basis(s, t));
        self.basis_cache.insert(key, Arc::clone(&basis));
        basis
    }

    fn signature_basis_key(
        &self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
    ) -> SignatureBasisKey {
        SignatureBasisKey {
            s,
            t,
            subalgebra: subalgebra.cache_key(),
            sig_index,
        }
    }

    fn signature_routing_key(
        &self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
    ) -> SignatureRoutingKey {
        SignatureRoutingKey {
            s,
            t,
            subalgebra: subalgebra.cache_key(),
        }
    }

    fn basis_signature_cached(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
    ) -> Arc<Vec<BasisElem>> {
        let key = self.signature_basis_key(s, t, subalgebra, sig_index);
        if let Some(cached) = self.basis_signature_cache.get(&key) {
            return Arc::clone(cached);
        }
        let gens_len = self.gens_by_s.get(s).map(Vec::len).unwrap_or(0);
        let mut pieces = Vec::with_capacity(gens_len);
        let mut estimated_len = 0;
        for index in 0..gens_len {
            let generator = self.gens_by_s[s][index];
            let generator_t = self.generators[generator].t;
            if generator_t > t {
                continue;
            }
            let alg_degree = t - generator_t;
            let coeffs = self.coeff_signature_basis_cached(alg_degree, subalgebra, sig_index);
            estimated_len += coeffs.len();
            pieces.push((generator, coeffs));
        }
        let mut sig_basis = Vec::with_capacity(estimated_len);
        for (generator, coeffs) in pieces {
            sig_basis.extend(coeffs.iter().map(|&coeff_packed| BasisElem {
                coeff_packed,
                generator,
            }));
        }
        let sig_basis = Arc::new(sig_basis);
        self.basis_signature_cache
            .insert(key, Arc::clone(&sig_basis));
        sig_basis
    }

    fn coeff_signature_basis_cached(
        &mut self,
        degree: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
    ) -> Arc<Vec<CoeffKey>> {
        let key = (subalgebra.cache_key(), degree, sig_index);
        if let Some(cached) = self.coeff_signature_basis_cache.get(&key) {
            return Arc::clone(cached);
        }
        let coeffs = self.algebra_basis[degree]
            .iter()
            .copied()
            .filter(|&coeff| subalgebra.signature_index_packed(coeff) == sig_index)
            .collect::<Vec<_>>();
        let coeffs = Arc::new(coeffs);
        self.coeff_signature_basis_cache
            .insert(key, Arc::clone(&coeffs));
        coeffs
    }

    fn basis_index_cached(&mut self, s: usize, t: usize) -> Arc<HashMap<(CoeffKey, usize), usize>> {
        let key = (s, t);
        if let Some(cached) = self.basis_index_cache.get(&key) {
            return Arc::clone(cached);
        }
        let basis = self.basis_cached(s, t);
        let index = Arc::new(basis_index(basis.as_slice()));
        self.basis_index_cache.insert(key, Arc::clone(&index));
        index
    }

    fn algebra_basis_index_cached(&mut self, degree: usize) -> Arc<HashMap<CoeffKey, usize>> {
        if let Some(cached) = self.algebra_basis_index_cache.get(&degree) {
            return Arc::clone(cached);
        }
        let index = Arc::new(self.build_algebra_basis_index(degree));
        self.algebra_basis_index_cache
            .insert(degree, Arc::clone(&index));
        index
    }

    fn build_algebra_basis_index(&self, degree: usize) -> HashMap<CoeffKey, usize> {
        self.algebra_basis[degree]
            .iter()
            .enumerate()
            .map(|(i, &coeff_packed)| (coeff_packed, i))
            .collect::<HashMap<_, _>>()
    }

    fn full_basis_lookup_cached(&mut self, s: usize, t: usize) -> Arc<FullBasisLookup> {
        let key = (s, t);
        if let Some(cached) = self.full_basis_lookup_cache.get(&key) {
            return Arc::clone(cached);
        }

        let mut generator_rows = vec![None; self.generators.len()];
        let mut offset = 0;
        let gens_len = self.gens_by_s.get(s).map(Vec::len).unwrap_or(0);
        for index in 0..gens_len {
            let generator = self.gens_by_s[s][index];
            let generator_t = self.generators[generator].t;
            if generator_t > t {
                continue;
            }
            let degree = t - generator_t;
            let coeff_index = self.algebra_basis_index_cached(degree);
            generator_rows[generator] = Some(GeneratorRows {
                offset,
                coeff_index,
            });
            offset += self.algebra_basis[degree].len();
        }

        let lookup = Arc::new(FullBasisLookup {
            generator_rows,
            target_dim: offset,
        });
        self.full_basis_lookup_cache
            .insert(key, Arc::clone(&lookup));
        lookup
    }

    fn basis_signature_index_cached(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
    ) -> Arc<HashMap<(CoeffKey, usize), usize>> {
        let key = self.signature_basis_key(s, t, subalgebra, sig_index);
        if let Some(cached) = self.basis_signature_index_cache.get(&key) {
            return Arc::clone(cached);
        }
        let basis = self.basis_signature_cached(s, t, subalgebra, sig_index);
        let index = Arc::new(basis_index(basis.as_slice()));
        self.basis_signature_index_cache
            .insert(key, Arc::clone(&index));
        index
    }

    fn basis_signature_routing_cached(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
    ) -> Arc<Vec<(usize, usize)>> {
        let key = self.signature_routing_key(s, t, subalgebra);
        if let Some(cached) = self.basis_signature_routing_cache.get(&key) {
            return Arc::clone(cached);
        }
        let full_basis = self.basis_cached(s, t);
        let mut counters = vec![0_usize; subalgebra.signatures().len()];
        let mut routing = Vec::with_capacity(full_basis.len());
        for elem in full_basis.iter() {
            let sig = subalgebra.signature_index_packed(elem.coeff_packed);
            let local = counters[sig];
            counters[sig] += 1;
            routing.push((sig, local));
        }
        let routing = Arc::new(routing);
        self.basis_signature_routing_cache
            .insert(key, Arc::clone(&routing));
        routing
    }

    fn signature_to_zero_translation(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
    ) -> Result<Arc<SignatureTranslation>, String> {
        let key = self.signature_basis_key(s, t, subalgebra, sig_index);
        if let Some(cached) = self.basis_signature_to_zero_cache.get(&key) {
            return Ok(Arc::clone(cached));
        }

        let sig_degree = subalgebra.signatures()[sig_index].degree();
        let shifted_t = t.checked_sub(sig_degree).ok_or_else(|| {
            format!(
                "signature {} has degree {sig_degree} above t={t}",
                subalgebra.signatures()[sig_index]
            )
        })?;
        let current_basis = self.basis_signature_cached(s, t, subalgebra, sig_index);
        let zero_basis = self.basis_signature_cached(s, shifted_t, subalgebra, 0);
        let current_len = current_basis.len();
        let zero_len = zero_basis.len();

        if current_len == zero_len {
            let mut identity = true;
            for (current, zero) in current_basis.iter().zip(zero_basis.iter()) {
                let Some((_signature, quotient)) =
                    subalgebra.split_profile_signature_packed(current.coeff_packed)
                else {
                    return Err(format!(
                        "subalgebra {} does not support profile signature translation",
                        subalgebra.name()
                    ));
                };
                if current.generator != zero.generator || quotient != zero.coeff_packed {
                    identity = false;
                    break;
                }
            }
            if identity {
                let translation = Arc::new(SignatureTranslation {
                    to_zero: Vec::new(),
                    from_zero: Vec::new(),
                    current_len,
                    zero_len,
                    identity: true,
                });
                self.basis_signature_to_zero_cache
                    .insert(key, Arc::clone(&translation));
                return Ok(translation);
            }
        }

        let zero_index = self.basis_signature_index_cached(s, shifted_t, subalgebra, 0);
        let mut to_zero = Vec::with_capacity(current_basis.len());
        let mut from_zero = vec![usize::MAX; zero_basis.len()];
        for (current, elem) in current_basis.iter().enumerate() {
            let Some((_signature, quotient)) =
                subalgebra.split_profile_signature_packed(elem.coeff_packed)
            else {
                return Err(format!(
                    "subalgebra {} does not support profile signature translation",
                    subalgebra.name()
                ));
            };
            let Some(&zero) = zero_index.get(&(quotient, elem.generator)) else {
                return Err(format!(
                    "internal error: signature quotient basis term missing for {} at (s,t)=({s},{t})",
                    subalgebra.name()
                ));
            };
            to_zero.push(zero);
            if from_zero[zero] != usize::MAX {
                return Err(format!(
                    "internal error: duplicate signature translation target for {} at (s,t)=({s},{t})",
                    subalgebra.name()
                ));
            }
            from_zero[zero] = current;
        }
        if from_zero.contains(&usize::MAX) {
            return Err(format!(
                "internal error: incomplete signature translation for {} at (s,t)=({s},{t})",
                subalgebra.name()
            ));
        }

        let translation = Arc::new(SignatureTranslation {
            to_zero,
            from_zero,
            current_len,
            zero_len,
            identity: false,
        });
        self.basis_signature_to_zero_cache
            .insert(key, Arc::clone(&translation));
        Ok(translation)
    }

    fn basis_signature_dim(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
    ) -> usize {
        let Some(gens_len) = self.gens_by_s.get(s).map(Vec::len) else {
            return 0;
        };
        let mut total = 0usize;
        for index in 0..gens_len {
            let generator_id = self.gens_by_s[s][index];
            let generator_t = self.generators[generator_id].t;
            if generator_t <= t {
                total += self
                    .coeff_signature_basis_cached(t - generator_t, subalgebra, sig_index)
                    .len();
            }
        }
        total
    }

    fn vector_to_terms(&self, s: usize, t: usize, vector: &BitVec) -> Vec<ModuleTerm> {
        let basis = self.basis(s, t);
        vector
            .ones()
            .map(|i| ModuleTerm {
                coeff_packed: basis[i].coeff_packed,
                generator: basis[i].generator,
            })
            .collect()
    }

    fn signature_vector_to_terms(
        &mut self,
        s: usize,
        t: usize,
        subalgebra: &Subalgebra,
        sig_index: usize,
        vector: &BitVec,
    ) -> Vec<ModuleTerm> {
        let basis = self.basis_signature_cached(s, t, subalgebra, sig_index);
        vector
            .ones()
            .map(|i| ModuleTerm {
                coeff_packed: basis[i].coeff_packed,
                generator: basis[i].generator,
            })
            .collect()
    }

    fn add_generator(&mut self, s: usize, t: usize, mut differential: Vec<ModuleTerm>) -> usize {
        while self.gens_by_s.len() <= s {
            self.gens_by_s.push(Vec::new());
        }
        sort_terms(&mut differential);
        let id = self.generators.len();
        self.generators.push(Generator {
            id,
            s,
            t,
            differential: Arc::new(differential),
        });
        self.gens_by_s[s].push(id);
        self.invalidate_basis_caches(s, t);
        id
    }

    fn verify_minimal_generator(&self, id: usize) -> Result<(), String> {
        let generator = &self.generators[id];
        for term in generator.differential.iter() {
            let target = self.generators.get(term.generator).ok_or_else(|| {
                format!(
                    "generator {} differential references missing generator {}",
                    self.generator_name(id),
                    term.generator
                )
            })?;
            let coeff_degree = Milnor::from_packed(term.coeff_packed).degree();
            if coeff_degree == 0 {
                return Err(format!(
                    "minimality violation: d({}) contains unit coefficient on {}",
                    self.generator_name(id),
                    self.generator_name(term.generator)
                ));
            }
            if target.t + coeff_degree != generator.t {
                return Err(format!(
                    "degree mismatch in d({}): coefficient degree {} plus target t={} does not equal source t={}",
                    self.generator_name(id),
                    coeff_degree,
                    target.t,
                    generator.t
                ));
            }
        }
        Ok(())
    }

    fn verify_generator_d_squared(&mut self, id: usize) -> Result<(), String> {
        let generator = self.generators[id].clone();
        if generator.s == 0 {
            return Ok(());
        }
        let d2 = self.differential_vector_from_terms(
            generator.s - 1,
            generator.t,
            &generator.differential,
        )?;
        if !d2.is_zero() {
            let remaining = self.format_vector(generator.s.saturating_sub(2), generator.t, &d2);
            return Err(format!(
                "d^2 verification failed for {}: {remaining}",
                self.generator_name(id)
            ));
        }
        Ok(())
    }

    fn invalidate_basis_caches(&mut self, s: usize, t: usize) {
        self.basis_cache
            .retain(|&(cached_s, cached_t), _| cached_s != s || cached_t < t);
        self.basis_index_cache
            .retain(|&(cached_s, cached_t), _| cached_s != s || cached_t < t);
        self.full_basis_lookup_cache
            .retain(|&(cached_s, cached_t), _| cached_s != s || cached_t < t);
        self.basis_signature_cache
            .retain(|key, _| key.s != s || key.t < t);
        self.basis_signature_index_cache
            .retain(|key, _| key.s != s || key.t < t);
        self.basis_signature_routing_cache
            .retain(|key, _| key.s != s || key.t < t);
        self.basis_signature_to_zero_cache
            .retain(|key, _| key.s != s || key.t < t);
        self.signature_matrix_cache
            .retain(|key, _| key.t < t || (key.s != s && key.s != s + 1));
        self.signature_solver_cache
            .retain(|key, _| key.t < t || (key.s != s && key.s != s + 1));
    }

    fn cached_multiply_packed_keys(
        &mut self,
        left_key: CoeffKey,
        right_key: CoeffKey,
    ) -> &[CoeffKey] {
        let key = (left_key, right_key);
        if !self.mult_packed_cache.contains_key(&key) {
            let value = multiply_packed_keys_with_row_cache(
                left_key,
                right_key,
                &mut self.row_decomp_cache,
            );
            self.mult_packed_cache.insert(key, value);
        }
        self.mult_packed_cache
            .get(&key)
            .expect("packed multiplication cache was just populated")
    }

    fn precompute_products_for_domain(
        &mut self,
        domain: &[BasisElem],
        subalgebra: Option<&Subalgebra>,
        s: usize,
        t: usize,
        sig_index: usize,
    ) {
        let timing_enabled = std::env::var_os("EXT_PROFILE_PRODUCT_TIMING").is_some();
        let total_timer = Instant::now();
        let profile_key = subalgebra.and_then(Subalgebra::profile_cache_key);
        let profile_enabled = std::env::var_os("EXT_PROFILE_MEMORY").is_some();
        let profile_verbose = std::env::var_os("EXT_PROFILE_MEMORY_VERBOSE").is_some();
        let estimated_pairs = domain
            .iter()
            .map(|elem| self.generators[elem.generator].differential.len())
            .sum::<usize>();
        let mut profile_product = profile_verbose || estimated_pairs >= 1_000_000;
        if profile_enabled && profile_product {
            log_process_memory(
                "product_precompute_start",
                format!(
                    "domain={} estimated_pairs={} profile_key={:?} mult_e0={} mult_e0_empty={} mult_e0_empty_segments={} mult_packed={}",
                    domain.len(),
                    estimated_pairs,
                    profile_key,
                    self.mult_e0_cache.len(),
                    self.mult_e0_empty_cache.len(),
                    self.mult_e0_empty_cache.segment_count(),
                    self.mult_packed_cache.len(),
                ),
            );
        }
        let missing_capacity = estimated_pairs.min(4096);
        let mut missing_full = Vec::<ProductKeyWithBound>::with_capacity(missing_capacity);
        let mut missing_e0 = Vec::<ProductKeyWithBound>::with_capacity(missing_capacity);
        if let Some(profile_key) = profile_key {
            self.activate_e0_product_profile(profile_key);
        }
        let scan_timer = Instant::now();
        // Skip keys already known to multiply to zero before they enter
        // missing_e0. This keeps the per-step sort/filter vector smaller
        // without adding another cache.
        let skip_cached_empty_during_scan = e0_scan_empty_skip_enabled()
            && profile_key.is_some()
            && self.mult_e0_empty_cache.supports_fast_membership_test()
            && !self.mult_e0_empty_cache.is_empty();
        let mut e0_scan_empty_hits = 0usize;
        for elem in domain {
            let raw_left_key = elem.coeff_packed;
            let left_key = if profile_key.is_some() {
                subalgebra
                    .expect("profile key requires a subalgebra")
                    .profile_quotient_packed_unchecked(raw_left_key)
            } else {
                raw_left_key
            };
            let generator = &self.generators[elem.generator];
            for term in generator.differential.iter() {
                let right_key = term.coeff_packed;
                if profile_key.is_some() {
                    let key = (left_key, right_key);
                    if self.mult_e0_cache.contains_key(&key) {
                        continue;
                    }
                    if skip_cached_empty_during_scan && self.mult_e0_empty_cache.contains(&key) {
                        e0_scan_empty_hits += 1;
                        continue;
                    }
                    let product_degree_bound = t
                        .checked_sub(self.generators[term.generator].t)
                        .expect("differential target generator degree should not exceed t");
                    missing_e0.push((left_key, right_key, product_degree_bound));
                } else {
                    let key = (left_key, right_key);
                    if !self.mult_packed_cache.contains_key(&key) {
                        let product_degree_bound = t
                            .checked_sub(self.generators[term.generator].t)
                            .expect("differential target generator degree should not exceed t");
                        missing_full.push((left_key, right_key, product_degree_bound));
                    }
                }
            }
        }
        let scan_s = scan_timer.elapsed().as_secs_f64();
        profile_product |= missing_e0.len() >= 1_000_000 || missing_full.len() >= 1_000_000;
        if profile_enabled && profile_product {
            log_process_memory(
                "product_after_scan",
                format!(
                    "domain={} estimated_pairs={} missing_full={} missing_full_mb={:.1} missing_e0={} missing_e0_mb={:.1}",
                    domain.len(),
                    estimated_pairs,
                    missing_full.len(),
                    bytes_to_mb(vec_product_key_bound_bytes(&missing_full)),
                    missing_e0.len(),
                    bytes_to_mb(vec_product_key_bound_bytes(&missing_e0)),
                ),
            );
        }

        let full_sort_timer = Instant::now();
        sort_dedup_product_key_bounds(&mut missing_full);
        let full_sort_s = full_sort_timer.elapsed().as_secs_f64();
        if profile_enabled && profile_product && !missing_full.is_empty() {
            log_process_memory(
                "product_after_full_dedup",
                format!(
                    "missing_full={} missing_full_mb={:.1}",
                    missing_full.len(),
                    bytes_to_mb(vec_product_key_bound_bytes(&missing_full)),
                ),
            );
        }
        let full_compute_timer = Instant::now();
        let computed_full = missing_full
            .par_iter()
            .map(|&(left_key, right_key, product_degree_bound)| {
                let product = multiply_packed_fast_bounded_matching(
                    left_key,
                    right_key,
                    product_degree_bound,
                    |_| true,
                );
                ((left_key, right_key), product)
            })
            .collect::<Vec<_>>();
        let full_compute_s = full_compute_timer.elapsed().as_secs_f64();

        let full_insert_timer = Instant::now();
        self.mult_packed_cache.reserve(computed_full.len());
        for (key, product) in computed_full {
            self.mult_packed_cache.entry(key).or_insert(product);
        }
        let full_insert_s = full_insert_timer.elapsed().as_secs_f64();

        let Some(subalgebra) = subalgebra else {
            if timing_enabled {
                eprintln!(
                    "ext_product_timing t={t} s={s} sig={sig_index} profile=none domain={} estimated_pairs={} scan_s={scan_s:.6} e0_scan_empty_hits={e0_scan_empty_hits} full_sort_s={full_sort_s:.6} full_compute_s={full_compute_s:.6} full_insert_s={full_insert_s:.6} total_s={:.6}",
                    domain.len(),
                    estimated_pairs,
                    total_timer.elapsed().as_secs_f64(),
                );
            }
            return;
        };
        let Some((_family, _n)) = profile_key else {
            if timing_enabled {
                eprintln!(
                    "ext_product_timing t={t} s={s} sig={sig_index} profile=no_key domain={} estimated_pairs={} scan_s={scan_s:.6} e0_scan_empty_hits={e0_scan_empty_hits} full_sort_s={full_sort_s:.6} full_compute_s={full_compute_s:.6} full_insert_s={full_insert_s:.6} total_s={:.6}",
                    domain.len(),
                    estimated_pairs,
                    total_timer.elapsed().as_secs_f64(),
                );
            }
            return;
        };
        let e0_sort_timer = Instant::now();
        sort_dedup_product_key_bounds(&mut missing_e0);
        let e0_sort_s = e0_sort_timer.elapsed().as_secs_f64();
        if profile_enabled && profile_product {
            log_process_memory(
                "product_after_e0_dedup",
                format!(
                    "missing_e0={} missing_e0_mb={:.1}",
                    missing_e0.len(),
                    bytes_to_mb(vec_product_key_bound_bytes(&missing_e0)),
                ),
            );
        }
        let prefilter_empty = self.should_prefilter_cached_e0_empty_products(missing_e0.len());
        let e0_filter_timer = Instant::now();
        if prefilter_empty {
            self.filter_cached_e0_empty_product_bounds(&mut missing_e0);
        }
        let e0_filter_s = e0_filter_timer.elapsed().as_secs_f64();
        let shrunk_missing_e0 = shrink_product_key_bound_vec_if_sparse(&mut missing_e0);
        if shrunk_missing_e0 {
            release_allocator_pressure();
        }
        if profile_enabled && profile_product {
            log_process_memory(
                "product_after_e0_empty_filter",
                format!(
                    "missing_e0={} missing_e0_mb={:.1} shrunk_missing_e0={} cached_empty={} cached_empty_segments={}",
                    missing_e0.len(),
                    bytes_to_mb(vec_product_key_bound_bytes(&missing_e0)),
                    shrunk_missing_e0,
                    self.mult_e0_empty_cache.len(),
                    self.mult_e0_empty_cache.segment_count(),
                ),
            );
        }
        let missing_len = missing_e0.len();
        let mut empty_write = 0;
        let mut chunk_start = 0;
        let e0_compute_timer = Instant::now();
        while chunk_start < missing_len {
            let chunk_end = (chunk_start + E0_PRECOMPUTE_CHUNK_PAIRS).min(missing_len);
            if profile_enabled && profile_product {
                log_process_memory(
                    "product_e0_chunk_start",
                    format!(
                        "chunk_start={} chunk_end={} chunk_len={} missing_e0_mb={:.1} empty_write={} mult_e0={} mult_e0_empty={}",
                        chunk_start,
                        chunk_end,
                        chunk_end - chunk_start,
                        bytes_to_mb(vec_product_key_bound_bytes(&missing_e0)),
                        empty_write,
                        self.mult_e0_cache.len(),
                        self.mult_e0_empty_cache.len(),
                    ),
                );
            }
            let computed_e0 = missing_e0[chunk_start..chunk_end]
                .par_iter()
                .map(|&(left_key, right_key, product_degree_bound)| {
                    let product = multiply_packed_fast_bounded_matching(
                        left_key,
                        right_key,
                        product_degree_bound,
                        |packed| subalgebra.profile_signature_is_zero_packed_unchecked(packed),
                    );
                    (
                        (left_key, right_key, product_degree_bound),
                        ProductList::from_vec(product),
                    )
                })
                .collect::<Vec<_>>();

            let nonempty_count = computed_e0
                .iter()
                .filter(|(_, product)| !product.is_empty())
                .count();
            if profile_enabled && profile_product {
                let computed_bytes = estimate_computed_e0_bound_bytes(&computed_e0);
                log_process_memory(
                    "product_e0_chunk_computed",
                    format!(
                        "chunk_start={} chunk_end={} computed={} nonempty={} computed_mb={:.1} empty_write={} mult_e0={}",
                        chunk_start,
                        chunk_end,
                        computed_e0.len(),
                        nonempty_count,
                        bytes_to_mb(computed_bytes),
                        empty_write,
                        self.mult_e0_cache.len(),
                    ),
                );
            }
            self.mult_e0_cache.reserve(nonempty_count);
            for ((left_key, right_key, product_degree_bound), product) in computed_e0 {
                if product.is_empty() {
                    missing_e0[empty_write] = (left_key, right_key, product_degree_bound);
                    empty_write += 1;
                } else {
                    self.mult_e0_cache
                        .entry((left_key, right_key))
                        .or_insert(product);
                }
            }
            if profile_enabled && profile_product {
                log_process_memory(
                    "product_e0_chunk_inserted",
                    format!(
                        "chunk_start={} chunk_end={} empty_write={} mult_e0={} mult_e0_terms={}",
                        chunk_start,
                        chunk_end,
                        empty_write,
                        self.mult_e0_cache.len(),
                        self.mult_e0_cache
                            .values()
                            .map(ProductList::len)
                            .sum::<usize>(),
                    ),
                );
            }
            chunk_start = chunk_end;
        }
        let e0_compute_s = e0_compute_timer.elapsed().as_secs_f64();
        missing_e0.truncate(empty_write);
        let e0_post_filter_timer = Instant::now();
        if !prefilter_empty {
            self.filter_cached_e0_empty_product_bounds(&mut missing_e0);
        }
        let e0_post_filter_s = e0_post_filter_timer.elapsed().as_secs_f64();
        if profile_enabled && profile_product {
            log_process_memory(
                "product_e0_before_merge_empty",
                format!(
                    "new_empty={} new_empty_mb={:.1} mult_e0={} cached_empty={} cached_empty_segments={}",
                    missing_e0.len(),
                    bytes_to_mb(vec_product_key_bound_bytes(&missing_e0)),
                    self.mult_e0_cache.len(),
                    self.mult_e0_empty_cache.len(),
                    self.mult_e0_empty_cache.segment_count(),
                ),
            );
        }
        release_allocator_pressure();
        let e0_merge_timer = Instant::now();
        let missing_e0_keys = missing_e0
            .into_iter()
            .map(|(left_key, right_key, _)| (left_key, right_key))
            .collect::<Vec<_>>();
        self.merge_e0_empty_products(missing_e0_keys);
        let e0_merge_s = e0_merge_timer.elapsed().as_secs_f64();
        release_allocator_pressure();
        if timing_enabled {
            eprintln!(
                "ext_product_timing t={t} s={s} sig={sig_index} profile={:?} domain={} estimated_pairs={} scan_s={scan_s:.6} e0_scan_empty_hits={e0_scan_empty_hits} full_sort_s={full_sort_s:.6} full_compute_s={full_compute_s:.6} full_insert_s={full_insert_s:.6} e0_sort_s={e0_sort_s:.6} e0_filter_s={e0_filter_s:.6} e0_post_filter_s={e0_post_filter_s:.6} e0_compute_s={e0_compute_s:.6} e0_merge_s={e0_merge_s:.6} prefilter_empty={} missing_len={} empty_write={} mult_e0={} mult_e0_empty={} total_s={:.6}",
                profile_key,
                domain.len(),
                estimated_pairs,
                prefilter_empty,
                missing_len,
                empty_write,
                self.mult_e0_cache.len(),
                self.mult_e0_empty_cache.len(),
                total_timer.elapsed().as_secs_f64(),
            );
        }
        if profile_enabled && profile_product {
            log_process_memory(
                "product_e0_after_merge_empty",
                format!(
                    "mult_e0={} mult_e0_empty={} mult_e0_empty_segments={} mult_e0_empty_mb={:.1}",
                    self.mult_e0_cache.len(),
                    self.mult_e0_empty_cache.len(),
                    self.mult_e0_empty_cache.segment_count(),
                    bytes_to_mb(self.mult_e0_empty_cache.estimated_bytes()),
                ),
            );
        }
    }

    fn format_vector(&self, s: usize, t: usize, vector: &BitVec) -> String {
        if vector.is_zero() {
            return "0".to_string();
        }
        let basis = self.basis(s, t);
        vector
            .ones()
            .map(|i| {
                let elem = &basis[i];
                if elem.coeff_packed == 0 {
                    self.generator_name(elem.generator)
                } else {
                    format!(
                        "{} {}",
                        Milnor::from_packed(elem.coeff_packed),
                        self.generator_name(elem.generator)
                    )
                }
            })
            .collect::<Vec<_>>()
            .join(" + ")
    }

    fn generator_name(&self, id: usize) -> String {
        let generator = &self.generators[id];
        if id == 0 {
            "g0".to_string()
        } else {
            format!("g{}_{}_{}", generator.s, generator.t, id)
        }
    }
}

fn fixed_t_basis_cache_survives_commit(
    cached_s: usize,
    cached_t: usize,
    commit_t: usize,
    invalidated_s: &[usize],
) -> bool {
    cached_t < commit_t || !invalidated_s.contains(&cached_s)
}

fn fixed_t_differential_cache_survives_commit(
    cached_s: usize,
    cached_t: usize,
    commit_t: usize,
    invalidated_s: &[usize],
) -> bool {
    cached_t < commit_t
        || invalidated_s
            .iter()
            .all(|&s| cached_s != s && cached_s != s.saturating_add(1))
}

fn filter_fixed_t_worker_caches_for_commit(
    caches: &mut FixedTBatchWorkerCaches,
    t: usize,
    invalidated_s: &[usize],
) {
    if invalidated_s.is_empty() {
        return;
    }

    caches.basis_cache.retain(|&(s, cached_t), _| {
        fixed_t_basis_cache_survives_commit(s, cached_t, t, invalidated_s)
    });
    caches.basis_index_cache.retain(|&(s, cached_t), _| {
        fixed_t_basis_cache_survives_commit(s, cached_t, t, invalidated_s)
    });
    caches.full_basis_lookup_cache.retain(|&(s, cached_t), _| {
        fixed_t_basis_cache_survives_commit(s, cached_t, t, invalidated_s)
    });
    caches
        .basis_signature_cache
        .retain(|key, _| fixed_t_basis_cache_survives_commit(key.s, key.t, t, invalidated_s));
    caches
        .basis_signature_index_cache
        .retain(|key, _| fixed_t_basis_cache_survives_commit(key.s, key.t, t, invalidated_s));
    caches
        .basis_signature_routing_cache
        .retain(|key, _| fixed_t_basis_cache_survives_commit(key.s, key.t, t, invalidated_s));
    caches
        .basis_signature_to_zero_cache
        .retain(|key, _| fixed_t_basis_cache_survives_commit(key.s, key.t, t, invalidated_s));
    caches.signature_matrix_cache.retain(|key, _| {
        fixed_t_differential_cache_survives_commit(key.s, key.t, t, invalidated_s)
    });
    caches.signature_solver_cache.retain(|key, _| {
        fixed_t_differential_cache_survives_commit(key.s, key.t, t, invalidated_s)
    });
}

fn filter_fixed_t_worker_caches_new_since(
    caches: &mut FixedTBatchWorkerCaches,
    baseline: &Resolution,
) {
    caches
        .basis_cache
        .retain(|key, _| !baseline.basis_cache.contains_key(key));
    caches
        .basis_index_cache
        .retain(|key, _| !baseline.basis_index_cache.contains_key(key));
    caches
        .algebra_basis_index_cache
        .retain(|key, _| !baseline.algebra_basis_index_cache.contains_key(key));
    caches
        .full_basis_lookup_cache
        .retain(|key, _| !baseline.full_basis_lookup_cache.contains_key(key));
    caches
        .coeff_signature_basis_cache
        .retain(|key, _| !baseline.coeff_signature_basis_cache.contains_key(key));
    caches
        .basis_signature_cache
        .retain(|key, _| !baseline.basis_signature_cache.contains_key(key));
    caches
        .basis_signature_index_cache
        .retain(|key, _| !baseline.basis_signature_index_cache.contains_key(key));
    caches
        .basis_signature_routing_cache
        .retain(|key, _| !baseline.basis_signature_routing_cache.contains_key(key));
    caches
        .basis_signature_to_zero_cache
        .retain(|key, _| !baseline.basis_signature_to_zero_cache.contains_key(key));
    caches
        .signature_matrix_cache
        .retain(|key, _| !baseline.signature_matrix_cache.contains_key(key));
    caches
        .signature_solver_cache
        .retain(|key, _| !baseline.signature_solver_cache.contains_key(key));
}

#[derive(Default)]
struct FixedTWorkerCacheOverlapStats {
    basis: usize,
    basis_index: usize,
    algebra_index: usize,
    full_lookup: usize,
    coeff_signature: usize,
    sig_basis: usize,
    sig_index: usize,
    sig_routing: usize,
    sig_translation: usize,
    sig_matrices: usize,
    sig_solvers: usize,
}

impl FixedTWorkerCacheOverlapStats {
    fn total(&self) -> usize {
        self.basis
            + self.basis_index
            + self.algebra_index
            + self.full_lookup
            + self.coeff_signature
            + self.sig_basis
            + self.sig_index
            + self.sig_routing
            + self.sig_translation
            + self.sig_matrices
            + self.sig_solvers
    }
}

fn dedup_map_by_key<K, V>(map: &mut HashMap<K, V>, seen: &mut FastHashSet<K>) -> usize
where
    K: Copy + Eq + std::hash::Hash,
{
    let before = map.len();
    map.retain(|key, _| seen.insert(*key));
    before - map.len()
}

fn dedup_fixed_t_worker_cache_overlaps(worker_caches: &mut [FixedTBatchWorkerCaches], t: usize) {
    let mut stats = FixedTWorkerCacheOverlapStats::default();

    macro_rules! dedup_field {
        ($field:ident, $stat:ident) => {{
            let mut seen = FastHashSet::default();
            for caches in worker_caches.iter_mut() {
                stats.$stat += dedup_map_by_key(&mut caches.$field, &mut seen);
            }
        }};
    }

    dedup_field!(basis_cache, basis);
    dedup_field!(basis_index_cache, basis_index);
    dedup_field!(algebra_basis_index_cache, algebra_index);
    dedup_field!(full_basis_lookup_cache, full_lookup);
    dedup_field!(coeff_signature_basis_cache, coeff_signature);
    dedup_field!(basis_signature_cache, sig_basis);
    dedup_field!(basis_signature_index_cache, sig_index);
    dedup_field!(basis_signature_routing_cache, sig_routing);
    dedup_field!(basis_signature_to_zero_cache, sig_translation);
    dedup_field!(signature_matrix_cache, sig_matrices);
    dedup_field!(signature_solver_cache, sig_solvers);

    if std::env::var_os("EXT_PROFILE_FIXED_T_WORKER_CACHE").is_some() {
        eprintln!(
            "fixed_t_worker_cache_overlap_dedup t={t} removed_total={} basis={} basis_index={} algebra_index={} full_lookup={} coeff_signature={} sig_basis={} sig_index={} sig_routing={} sig_translation={} sig_matrices={} sig_solvers={}",
            stats.total(),
            stats.basis,
            stats.basis_index,
            stats.algebra_index,
            stats.full_lookup,
            stats.coeff_signature,
            stats.sig_basis,
            stats.sig_index,
            stats.sig_routing,
            stats.sig_translation,
            stats.sig_matrices,
            stats.sig_solvers,
        );
    }
}

fn log_fixed_t_worker_cache_delta_profile(
    caches: &FixedTBatchWorkerCaches,
    t: usize,
    group: &[usize],
) {
    if std::env::var_os("EXT_PROFILE_FIXED_T_WORKER_CACHE").is_none() {
        return;
    }
    let matrix_bytes = caches
        .signature_matrix_cache
        .values()
        .map(|matrix| matrix.estimated_bytes())
        .sum::<usize>();
    let solver_bytes = caches
        .signature_solver_cache
        .values()
        .map(|solver| solver.estimated_bytes())
        .sum::<usize>();
    let group_first = group.first().copied().unwrap_or(0);
    let group_last = group.last().copied().unwrap_or(0);
    eprintln!(
        "fixed_t_batch_worker_cache_delta t={t} group_first={group_first} group_last={group_last} group_len={} basis={} basis_mb={:.1} basis_index={} basis_index_mb={:.1} full_lookup={} full_lookup_mb={:.1} sig_basis={} sig_basis_mb={:.1} sig_matrices={} sig_matrix_mb={:.1} sig_solvers={} sig_solver_mb={:.1}",
        group.len(),
        caches.basis_cache.len(),
        bytes_to_mb(estimate_arc_vec_cache(&caches.basis_cache)),
        caches.basis_index_cache.len(),
        bytes_to_mb(estimate_arc_map_cache(&caches.basis_index_cache)),
        caches.full_basis_lookup_cache.len(),
        bytes_to_mb(
            caches
                .full_basis_lookup_cache
                .values()
                .map(|lookup| lookup.estimated_bytes())
                .sum::<usize>()
        ),
        caches.basis_signature_cache.len(),
        bytes_to_mb(estimate_arc_vec_cache(&caches.basis_signature_cache)),
        caches.signature_matrix_cache.len(),
        bytes_to_mb(matrix_bytes),
        caches.signature_solver_cache.len(),
        bytes_to_mb(solver_bytes),
    );
}

fn log_fixed_t_group_done_profile(
    t: usize,
    scheduler: FixedTBatchScheduler,
    group: &FixedTBatchGroupResult,
) {
    if std::env::var_os("EXT_PROFILE_FIXED_T_GROUPS").is_none() {
        return;
    }
    eprintln!(
        "fixed_t_batch_group_done t={t} scheduler={} group_index={} group_first={} group_last={} group_len={} wall_s={:.6} task_sum_s={:.6} task_max_s={:.6}",
        scheduler.as_str(),
        group.group_index,
        group.group_first,
        group.group_last,
        group.group_len,
        group.wall_s,
        group.task_sum_s,
        group.task_max_s,
    );
}

fn log_fixed_t_group_worker_done_profile(t: usize, group: &FixedTBatchGroupResult) {
    if std::env::var_os("EXT_PROFILE_FIXED_T_GROUPS").is_none() {
        return;
    }
    eprintln!(
        "fixed_t_batch_group_worker_done t={t} group_index={} group_first={} group_last={} group_len={} wall_s={:.6} task_sum_s={:.6} task_max_s={:.6}",
        group.group_index,
        group.group_first,
        group.group_last,
        group.group_len,
        group.wall_s,
        group.task_sum_s,
        group.task_max_s,
    );
}

fn fixed_t_detailed_logging_enabled() -> bool {
    std::env::var_os("EXT_PROFILE_FIXED_T_TASKS").is_some()
}

fn fixed_t_worker_product_cache_limit_bytes() -> Option<usize> {
    let Some(raw) = std::env::var("EXT_FIXED_T_WORKER_PRODUCT_CACHE_MB").ok() else {
        return DEFAULT_FIXED_T_WORKER_PRODUCT_CACHE_MB.checked_mul(1024 * 1024);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "0" {
        return None;
    }
    match trimmed.parse::<usize>() {
        Ok(limit_mb) if limit_mb > 0 => limit_mb.checked_mul(1024 * 1024),
        _ => {
            eprintln!(
                "warning: ignoring invalid EXT_FIXED_T_WORKER_PRODUCT_CACHE_MB={raw:?}; expected a positive integer MiB limit"
            );
            None
        }
    }
}

fn fixed_t_e0_profile_bank_enabled() -> bool {
    let Some(raw) = std::env::var("EXT_FIXED_T_E0_PROFILE_BANK").ok() else {
        return true;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return false;
    }
    if matches!(trimmed, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON") {
        return true;
    }
    if matches!(
        trimmed,
        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
    ) {
        return false;
    }
    eprintln!(
        "warning: ignoring invalid EXT_FIXED_T_E0_PROFILE_BANK={raw:?}; expected 0/1, true/false, yes/no, or on/off"
    );
    false
}

fn fixed_t_choice_dim_prewarm_enabled() -> bool {
    let Some(raw) = std::env::var("EXT_FIXED_T_CHOICE_DIM_PREWARM").ok() else {
        return true;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return false;
    }
    if matches!(trimmed, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON") {
        return true;
    }
    if matches!(
        trimmed,
        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
    ) {
        return false;
    }
    eprintln!(
        "warning: ignoring invalid EXT_FIXED_T_CHOICE_DIM_PREWARM={raw:?}; expected 0/1, true/false, yes/no, or on/off"
    );
    false
}

fn fixed_t_auto_subalgebras(mode: &ComputeMode) -> Option<(&[Subalgebra], bool)> {
    match mode {
        ComputeMode::Auto {
            subalgebras, force, ..
        }
        | ComputeMode::AutoBoundedNaiveFallback { subalgebras, force } => {
            Some((subalgebras.as_slice(), *force))
        }
        _ => None,
    }
}

fn estimate_fixed_t_task_weight_from_dims(s: usize, basis_dims: &[usize]) -> u128 {
    let domain = basis_dims[s] as u128;
    let target = if s == 0 { 0 } else { basis_dims[s - 1] as u128 };
    let boundary_domain = basis_dims.get(s + 1).copied().unwrap_or(0) as u128;
    estimate_fixed_t_task_weight_from_dimensions(domain, target, boundary_domain)
}

fn estimate_fixed_t_task_weight_from_dimensions(
    domain: impl TryInto<u128>,
    target: impl TryInto<u128>,
    boundary_domain: impl TryInto<u128>,
) -> u128 {
    let domain = domain.try_into().ok().unwrap_or(u128::MAX);
    let target = target.try_into().ok().unwrap_or(u128::MAX);
    let boundary_domain = boundary_domain.try_into().ok().unwrap_or(u128::MAX);
    let word_target = target.div_ceil(64).max(1);
    let word_domain = domain.div_ceil(64).max(1);
    let matrix_work = domain.saturating_mul(word_target);
    let boundary_work = boundary_domain.saturating_mul(word_domain);
    matrix_work
        .saturating_add(boundary_work)
        .saturating_add(domain)
        .saturating_add(boundary_domain)
        .max(1)
}

fn fixed_t_subalgebra_build_cost_multiplier(subalgebra: &Subalgebra) -> (u128, u128) {
    // Central place to calibrate per-subalgebra matrix-construction constants c_B.
    let name = subalgebra.name();
    let numerator = if name == "A3" {
        15
    } else if name == "B3221" {
        5
    } else if name.starts_with('B') {
        3
    } else if name == "A2" {
        2
    } else {
        1
    };
    (numerator, 1)
}

fn apply_fixed_t_subalgebra_build_cost_multiplier(weight: u128, subalgebra: &Subalgebra) -> u128 {
    let (numerator, denominator) = fixed_t_subalgebra_build_cost_multiplier(subalgebra);
    let numerator = numerator.max(1);
    let denominator = denominator.max(1);
    weight
        .saturating_mul(numerator)
        .div_ceil(denominator)
        .max(1)
}

fn fixed_t_dimension_task_weights(active_s: &[usize], basis_dims: &[usize]) -> Vec<u128> {
    active_s
        .iter()
        .map(|&s| estimate_fixed_t_task_weight_from_dims(s, basis_dims))
        .collect()
}

fn fixed_t_task_dimensions_from_dims(
    s: usize,
    t: usize,
    basis_dims: &[usize],
) -> (usize, usize, usize, usize) {
    let domain_dim = basis_dims[s];
    let target_dim = if s == 0 {
        usize::from(t == 0)
    } else {
        basis_dims[s - 1]
    };
    let boundary_domain_dim = basis_dims.get(s + 1).copied().unwrap_or(0);
    (domain_dim, target_dim, boundary_domain_dim, domain_dim)
}

fn empty_fixed_t_batch_task_result_from_dims(
    s: usize,
    t: usize,
    basis_dims: &[usize],
) -> FixedTBatchTaskResult {
    let d_domain_dim = fixed_t_task_dimensions_from_dims(s, t, basis_dims).0;
    debug_assert_eq!(d_domain_dim, 0);
    fixed_t_batch_skip_task_result_from_dims(s, t, basis_dims, FIXED_T_BATCH_SKIP_EMPTY_DOMAIN_PATH)
}

fn adams_vanishing_fixed_t_batch_task_result_from_dims(
    s: usize,
    t: usize,
    basis_dims: &[usize],
) -> FixedTBatchTaskResult {
    debug_assert!(fixed_t_task_is_adams_vanishing(s, t));
    fixed_t_batch_skip_task_result_from_dims(
        s,
        t,
        basis_dims,
        FIXED_T_BATCH_SKIP_ADAMS_VANISHING_PATH,
    )
}

fn fixed_t_batch_skip_task_result_from_dims(
    s: usize,
    t: usize,
    basis_dims: &[usize],
    path: &str,
) -> FixedTBatchTaskResult {
    let (d_domain_dim, d_target_dim, boundary_domain_dim, boundary_target_dim) =
        fixed_t_task_dimensions_from_dims(s, t, basis_dims);
    FixedTBatchTaskResult {
        s,
        output_q: s + 1,
        t,
        path: path.to_string(),
        representatives: Vec::new(),
        d_domain_dim,
        d_target_dim,
        boundary_domain_dim,
        boundary_target_dim,
        reduced_domain_dim: None,
        reduced_target_dim: None,
        reduced_boundary_domain_dim: None,
        candidate_subalgebras: Vec::new(),
        chosen_subalgebra: None,
        matrix_s: 0.0,
        homology_s: 0.0,
        total_s: 0.0,
    }
}

fn fixed_t_task_path_is_trivial_skip(path: &str) -> bool {
    matches!(
        path,
        FIXED_T_BATCH_SKIP_EMPTY_DOMAIN_PATH | FIXED_T_BATCH_SKIP_ADAMS_VANISHING_PATH
    )
}

fn fixed_t_task_result_is_trivial_skip(result: &FixedTBatchTaskResult) -> bool {
    fixed_t_task_path_is_trivial_skip(&result.path)
}

fn fixed_t_task_result_path_count(results: &[FixedTBatchTaskResult], path: &str) -> usize {
    results.iter().filter(|result| result.path == path).count()
}

fn fixed_t_task_result_path_bounds<'a>(
    results: &'a [FixedTBatchTaskResult],
    path: &str,
) -> (
    Option<&'a FixedTBatchTaskResult>,
    Option<&'a FixedTBatchTaskResult>,
) {
    let mut matching = results.iter().filter(|result| result.path == path);
    let first = matching.next();
    let last = matching.next_back().or(first);
    (first, last)
}

fn verify_minimal_terms_against_snapshot(
    t: usize,
    differential: &[ModuleTerm],
    view: &FrozenResolutionView,
) -> Result<(), String> {
    for term in differential {
        let target = view.generator(term.generator);
        let coeff_degree = Milnor::from_packed(term.coeff_packed).degree();
        if coeff_degree == 0 {
            return Err(format!(
                "fixed-t-batch minimality violation at t={t}: representative uses unit coefficient on g{}",
                term.generator
            ));
        }
        if target.t + coeff_degree != t {
            return Err(format!(
                "fixed-t-batch degree mismatch at t={t}: coefficient degree {} plus target generator t={} for g{}",
                coeff_degree, target.t, term.generator
            ));
        }
        if target.t >= t {
            return Err(format!(
                "fixed-t-batch frozen view violation at t={t}: representative references same-or-higher degree generator g{} with t={}",
                term.generator, target.t
            ));
        }
    }
    Ok(())
}

fn compare_fixed_t_shadow(
    t: usize,
    sequential: &Resolution,
    batch: &Resolution,
) -> Result<(), String> {
    let sequential_counts = bidegree_counts_at_t(sequential, t);
    let batch_counts = bidegree_counts_at_t(batch, t);
    if sequential_counts != batch_counts {
        return Err(format!(
            "fixed-t-batch-shadow mismatch at t={t}: sequential={sequential_counts:?} batch={batch_counts:?}"
        ));
    }
    Ok(())
}

fn candidate_subalgebra_is_better(
    candidate: &Subalgebra,
    candidate_key: (usize, usize),
    incumbent: &Subalgebra,
    incumbent_key: (usize, usize),
) -> bool {
    if let Some(candidate_wins) = f_over_matching_fp_preference(candidate, incumbent) {
        return candidate_wins;
    }
    candidate_key < incumbent_key
}

fn subalgebra_selection_is_disabled(subalgebra: &Subalgebra) -> bool {
    matches!(subalgebra.f_family_index(), Some((3, false)))
}

fn f_over_matching_fp_preference(candidate: &Subalgebra, incumbent: &Subalgebra) -> Option<bool> {
    let (candidate_n, candidate_is_f) = candidate.f_family_index()?;
    let (incumbent_n, incumbent_is_f) = incumbent.f_family_index()?;
    if candidate_n == incumbent_n
        && matches!(candidate_n, 1 | 2)
        && candidate_is_f != incumbent_is_f
    {
        Some(candidate_is_f)
    } else {
        None
    }
}

fn algorithm2_method_label(subalgebra: &Subalgebra, s: usize, t: usize) -> String {
    if subalgebra_selection_mode() == SubalgebraSelectionMode::Detailed {
        let applicability = subalgebra.selection_applicability(s, t);
        format!(
            "algorithm2({};source={})",
            subalgebra.name(),
            applicability.certification_source
        )
    } else {
        format!("algorithm2({})", subalgebra.name())
    }
}

fn fixed_t_batch_inner_name(mode: &ComputeMode) -> String {
    match mode {
        ComputeMode::Naive => "naive".to_string(),
        ComputeMode::Auto {
            subalgebras,
            strict,
            force,
        } => format!(
            "auto({};strict={strict};force={force})",
            subalgebras
                .iter()
                .map(Subalgebra::name)
                .collect::<Vec<_>>()
                .join(",")
        ),
        ComputeMode::AutoBoundedNaiveFallback { subalgebras, force } => format!(
            "auto-bounded-naive({};force={force})",
            subalgebras
                .iter()
                .map(Subalgebra::name)
                .collect::<Vec<_>>()
                .join(",")
        ),
        ComputeMode::Accelerated {
            subalgebra,
            strict,
            force,
        } => format!(
            "algorithm2({};strict={strict};force={force})",
            subalgebra.name()
        ),
        ComputeMode::FixedTBatch { .. } => "fixed-t-batch".to_string(),
    }
}

fn contiguous_groups(values: &[usize], group_count: usize) -> Vec<Vec<usize>> {
    let group_count = group_count.max(1).min(values.len().max(1));
    let chunk_size = values.len().div_ceil(group_count);
    values
        .chunks(chunk_size.max(1))
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn round_robin_groups(values: &[usize], group_count: usize) -> Vec<Vec<usize>> {
    let mut groups = round_robin_rank_groups(values, group_count);
    groups.retain(|group| !group.is_empty());
    groups
}

fn round_robin_rank_groups(values: &[usize], group_count: usize) -> Vec<Vec<usize>> {
    let group_count = group_count.max(1);
    let mut groups = (0..group_count).map(|_| Vec::new()).collect::<Vec<_>>();
    for &value in values {
        groups[value % group_count].push(value);
    }
    groups
}

#[cfg(test)]
fn fixed_t_local_s_groups(
    values: &[usize],
    basis_dims: &[usize],
    local_s_workers: usize,
) -> Vec<Vec<usize>> {
    if values.is_empty() {
        return Vec::new();
    }
    let group_count = local_s_workers.max(1).min(values.len());
    if group_count == 1 {
        return vec![values.to_vec()];
    }
    if group_count == values.len() {
        return values.iter().map(|&value| vec![value]).collect();
    }
    let weights = fixed_t_dimension_task_weights(values, basis_dims);
    minimax_contiguous_groups(values, &weights, group_count)
}

#[cfg(test)]
fn fixed_t_local_s_chunk_count(value_count: usize, _local_s_workers: usize) -> usize {
    value_count
}

#[cfg(test)]
fn fixed_t_local_s_chunks(
    values: &[usize],
    _basis_dims: &[usize],
    _local_s_workers: usize,
    _t: usize,
) -> Vec<Vec<usize>> {
    values.iter().map(|&value| vec![value]).collect()
}

#[cfg(test)]
fn group_weight(group: &[usize], values: &[usize], weights: &[u128]) -> u128 {
    group
        .iter()
        .map(|&value| weight_for_value(value, values, weights))
        .fold(0_u128, |sum, weight| sum.saturating_add(weight))
}

fn weight_for_value(value: usize, values: &[usize], weights: &[u128]) -> u128 {
    values
        .binary_search(&value)
        .ok()
        .and_then(|index| weights.get(index).copied())
        .unwrap_or(1)
}

fn weighted_contiguous_groups(
    values: &[usize],
    weights: &[u128],
    group_count: usize,
) -> Vec<Vec<usize>> {
    assert_eq!(
        values.len(),
        weights.len(),
        "weighted_contiguous_groups requires one weight per value"
    );
    if values.is_empty() {
        return Vec::new();
    }
    let group_count = group_count.max(1).min(values.len());
    if group_count == 1 {
        return vec![values.to_vec()];
    }

    let mut remaining_groups = group_count;
    let mut remaining_weight = weights.iter().copied().sum::<u128>();
    let mut groups = Vec::with_capacity(group_count);
    let mut current = Vec::new();
    let mut current_weight = 0u128;

    for (index, (&value, &weight)) in values.iter().zip(weights).enumerate() {
        current.push(value);
        current_weight = current_weight.saturating_add(weight);
        let remaining_values = values.len() - index - 1;
        if remaining_groups > 1 && remaining_values >= remaining_groups - 1 {
            let must_split_to_leave_nonempty_groups = remaining_values == remaining_groups - 1;
            let should_split_by_weight = if remaining_weight == 0 {
                let remaining_slots = current.len() + remaining_values;
                let target_len = remaining_slots.div_ceil(remaining_groups);
                current.len() >= target_len.max(1)
            } else {
                let target = remaining_weight.div_ceil(remaining_groups as u128).max(1);
                current_weight >= target
            };
            if must_split_to_leave_nonempty_groups || should_split_by_weight {
                remaining_weight = remaining_weight.saturating_sub(current_weight);
                remaining_groups -= 1;
                groups.push(std::mem::take(&mut current));
                current_weight = 0;
            }
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    debug_assert_eq!(groups.len(), group_count);
    debug_assert_eq!(
        groups.iter().map(Vec::len).sum::<usize>(),
        values.len(),
        "weighted contiguous grouping lost values"
    );
    groups
}

fn low_prefix_build_cost_groups(
    values: &[usize],
    weights: &[u128],
    group_count: usize,
) -> Vec<Vec<usize>> {
    debug_assert_eq!(values.len(), weights.len());
    let Some(&last) = values.last() else {
        return Vec::new();
    };
    if group_count != 4 {
        return weighted_contiguous_groups(values, weights, group_count);
    }

    let cuts: &[usize] = if last >= 32 {
        &[4, 9, 18]
    } else if last >= 12 {
        &[4, 7, 10]
    } else {
        return weighted_contiguous_groups(values, weights, group_count);
    };

    let mut groups = Vec::with_capacity(4);
    let mut start_index = 0usize;
    for &cut in cuts {
        let end_index = values.partition_point(|&value| value <= cut);
        if end_index == start_index {
            return weighted_contiguous_groups(values, weights, group_count);
        }
        groups.push(values[start_index..end_index].to_vec());
        start_index = end_index;
    }
    if start_index >= values.len() {
        return weighted_contiguous_groups(values, weights, group_count);
    }
    groups.push(values[start_index..].to_vec());
    groups
}

fn weighted_contiguous_front_merge_tail_split_groups(
    values: &[usize],
    weights: &[u128],
    group_count: usize,
) -> Vec<Vec<usize>> {
    debug_assert_eq!(values.len(), weights.len());
    let baseline = weighted_contiguous_groups(values, weights, group_count);
    if baseline.len() < 3 {
        return baseline;
    }

    let mut out = Vec::with_capacity(baseline.len());
    let mut first = baseline[0].clone();
    first.extend_from_slice(&baseline[1]);
    out.push(first);

    for group in baseline.iter().take(baseline.len() - 1).skip(2) {
        out.push(group.clone());
    }

    let tail = baseline.last().expect("baseline has at least three groups");
    if tail.len() <= 1 {
        out.push(tail.clone());
        return out;
    }
    let tail_weights = tail
        .iter()
        .map(|&s| weight_for_value(s, values, weights).max(1))
        .collect::<Vec<_>>();
    out.extend(weighted_contiguous_tail_split_two_to_one(
        tail,
        &tail_weights,
    ));
    out
}

fn weighted_contiguous_tail_split_two_to_one(
    values: &[usize],
    weights: &[u128],
) -> Vec<Vec<usize>> {
    debug_assert_eq!(values.len(), weights.len());
    if values.len() <= 1 {
        return vec![values.to_vec()];
    }

    let weights = weights
        .iter()
        .map(|&weight| weight.max(1))
        .collect::<Vec<_>>();
    let total = weights
        .iter()
        .copied()
        .fold(0_u128, |sum, weight| sum.saturating_add(weight));
    if total == 0 {
        let split = values.len().saturating_sub(1).max(1);
        return vec![values[..split].to_vec(), values[split..].to_vec()];
    }

    let mut prefix = 0_u128;
    let mut best_split = 1_usize;
    let mut best_score: Option<u128> = None;
    for split in 1..values.len() {
        prefix = prefix.saturating_add(weights[split - 1]);
        let left = prefix;
        let right = total.saturating_sub(left);
        let left_scaled = left.saturating_mul(1);
        let right_scaled = right.saturating_mul(2);
        let score = left_scaled.abs_diff(right_scaled);
        if best_score.is_none_or(|best| score < best) {
            best_score = Some(score);
            best_split = split;
        }
    }

    vec![values[..best_split].to_vec(), values[best_split..].to_vec()]
}

fn sticky_contiguous_groups(
    values: &[usize],
    weights: &[u128],
    group_count: usize,
    previous_ranges: Option<&[(usize, usize)]>,
    max_over_base_num: u128,
    max_over_base_den: u128,
    t: usize,
) -> Vec<Vec<usize>> {
    debug_assert_eq!(values.len(), weights.len());
    let baseline = weighted_contiguous_groups(values, weights, group_count);
    let Some(previous_ranges) = previous_ranges else {
        return baseline;
    };
    let Some(sticky) = sticky_groups_from_previous_ranges(values, previous_ranges, group_count)
    else {
        if std::env::var_os("EXT_PROFILE_FIXED_T_GROUPS").is_some() {
            eprintln!(
                "fixed_t_sticky_scheduler t={t} previous_ranges={} chosen=baseline reason=incompatible-ranges",
                previous_ranges.len(),
            );
        }
        return baseline;
    };

    let base_max = max_group_weight(&baseline, values, weights).max(1);
    let sticky_max = max_group_weight(&sticky, values, weights).max(1);
    let allowed = base_max
        .saturating_mul(max_over_base_num.max(1))
        .div_ceil(max_over_base_den.max(1));
    let use_sticky = sticky_max <= allowed;
    if std::env::var_os("EXT_PROFILE_FIXED_T_GROUPS").is_some() {
        eprintln!(
            "fixed_t_sticky_scheduler t={t} previous_ranges={} baseline_groups={} sticky_groups={} base_max_weight={} sticky_max_weight={} allowed_max_weight={} chosen={}",
            previous_ranges.len(),
            baseline.len(),
            sticky.len(),
            base_max,
            sticky_max,
            allowed,
            if use_sticky { "sticky" } else { "baseline" },
        );
    }
    if use_sticky { sticky } else { baseline }
}

fn sticky_groups_from_previous_ranges(
    values: &[usize],
    previous_ranges: &[(usize, usize)],
    group_count: usize,
) -> Option<Vec<Vec<usize>>> {
    if values.is_empty() {
        return Some(Vec::new());
    }
    let group_count = group_count.max(1).min(values.len());
    let active_first = *values.first()?;
    let active_last = *values.last()?;
    let mut ranges = previous_ranges
        .iter()
        .copied()
        .filter(|(first, last)| first <= last)
        .collect::<Vec<_>>();
    ranges.sort_unstable();
    ranges.dedup();

    let mut starts = Vec::with_capacity(group_count);
    starts.push(active_first);
    for (first, _) in ranges {
        if first <= active_first || first > active_last {
            continue;
        }
        let index = lower_bound(values, first);
        if index < values.len() {
            let start = values[index];
            if starts.last().copied() != Some(start) {
                starts.push(start);
            }
        }
    }
    if starts.len() != group_count {
        return None;
    }

    let mut groups = Vec::with_capacity(group_count);
    for (index, &start) in starts.iter().enumerate() {
        let start_index = lower_bound(values, start);
        let end_index = starts
            .get(index + 1)
            .map(|&next_start| lower_bound(values, next_start))
            .unwrap_or(values.len());
        if start_index >= end_index {
            return None;
        }
        groups.push(values[start_index..end_index].to_vec());
    }
    Some(groups)
}

fn lower_bound(values: &[usize], needle: usize) -> usize {
    values.partition_point(|&value| value < needle)
}

fn max_group_weight(groups: &[Vec<usize>], values: &[usize], weights: &[u128]) -> u128 {
    groups
        .iter()
        .map(|group| {
            group
                .iter()
                .filter_map(|&value| {
                    values
                        .binary_search(&value)
                        .ok()
                        .map(|index| weights[index].max(1))
                })
                .fold(0_u128, |sum, weight| sum.saturating_add(weight))
        })
        .max()
        .unwrap_or(0)
}

fn minimax_contiguous_groups(
    values: &[usize],
    weights: &[u128],
    group_count: usize,
) -> Vec<Vec<usize>> {
    debug_assert_eq!(values.len(), weights.len());
    if values.is_empty() {
        return Vec::new();
    }
    let group_count = group_count.max(1).min(values.len());
    if group_count == 1 {
        return vec![values.to_vec()];
    }

    let normalized = weights
        .iter()
        .map(|&weight| weight.max(1))
        .collect::<Vec<_>>();
    let mut lower = normalized.iter().copied().max().unwrap_or(1);
    let mut upper = normalized
        .iter()
        .copied()
        .fold(0_u128, |sum, weight| sum.saturating_add(weight));
    while lower < upper {
        let mid = lower + (upper - lower) / 2;
        if can_partition_contiguous_with_max(&normalized, group_count, mid) {
            upper = mid;
        } else {
            lower = mid + 1;
        }
    }
    reconstruct_minimax_contiguous_groups(values, &normalized, group_count, lower)
}

fn can_partition_contiguous_with_max(
    weights: &[u128],
    group_count: usize,
    max_group_weight: u128,
) -> bool {
    let mut groups = 1usize;
    let mut current = 0u128;
    for &weight in weights {
        if weight > max_group_weight {
            return false;
        }
        if current.saturating_add(weight) > max_group_weight {
            groups += 1;
            current = weight;
            if groups > group_count {
                return false;
            }
        } else {
            current += weight;
        }
    }
    true
}

fn reconstruct_minimax_contiguous_groups(
    values: &[usize],
    weights: &[u128],
    group_count: usize,
    max_group_weight: u128,
) -> Vec<Vec<usize>> {
    let mut groups = Vec::with_capacity(group_count);
    let mut end = values.len();
    let mut remaining_groups = group_count;
    while remaining_groups > 0 {
        let min_start = remaining_groups - 1;
        let mut start = end;
        let mut current = 0u128;
        while start > min_start {
            let weight = weights[start - 1];
            if start < end && current.saturating_add(weight) > max_group_weight {
                break;
            }
            start -= 1;
            current = current.saturating_add(weight);
        }
        groups.push(values[start..end].to_vec());
        end = start;
        remaining_groups -= 1;
    }
    groups.reverse();
    groups
}

fn profile_contiguous_groups<P>(
    values: &[usize],
    weights: &[u128],
    profiles: &[P],
    group_count: usize,
) -> Vec<Vec<usize>>
where
    P: Eq,
{
    debug_assert_eq!(values.len(), weights.len());
    debug_assert_eq!(values.len(), profiles.len());
    if values.is_empty() {
        return Vec::new();
    }
    let group_count = group_count.max(1).min(values.len());
    if group_count == 1 {
        return vec![values.to_vec()];
    }

    let mut runs = Vec::<(usize, usize, u128)>::new();
    let mut start = 0usize;
    while start < values.len() {
        let mut end = start + 1;
        while end < values.len() && profiles[end] == profiles[start] {
            end += 1;
        }
        let weight = weights[start..end]
            .iter()
            .fold(0_u128, |sum, &weight| sum.saturating_add(weight.max(1)));
        runs.push((start, end, weight));
        start = end;
    }

    let mut allocations = runs.iter().map(|_| 1usize).collect::<Vec<_>>();

    while allocations.iter().sum::<usize>() > group_count {
        merge_lightest_adjacent_profile_runs(&mut runs, &mut allocations);
    }
    while allocations.iter().sum::<usize>() < group_count {
        if !split_heaviest_profile_run(&runs, &mut allocations) {
            break;
        }
    }

    let mut groups = Vec::with_capacity(group_count);
    for ((start, end, _), allocation) in runs.into_iter().zip(allocations) {
        groups.extend(weighted_contiguous_groups(
            &values[start..end],
            &weights[start..end],
            allocation,
        ));
    }
    groups
}

fn merge_lightest_adjacent_profile_runs(
    runs: &mut Vec<(usize, usize, u128)>,
    allocations: &mut Vec<usize>,
) {
    let merge_index = runs
        .windows(2)
        .enumerate()
        .min_by(|(left_index, left), (right_index, right)| {
            let left_weight = left[0].2.saturating_add(left[1].2);
            let right_weight = right[0].2.saturating_add(right[1].2);
            left_weight
                .cmp(&right_weight)
                .then_with(|| left_index.cmp(right_index))
        })
        .map(|(index, _)| index)
        .expect("at least two profile runs are available to merge");
    let merged = (
        runs[merge_index].0,
        runs[merge_index + 1].1,
        runs[merge_index].2.saturating_add(runs[merge_index + 1].2),
    );
    let merged_allocation = allocations[merge_index]
        .saturating_add(allocations[merge_index + 1])
        .saturating_sub(1)
        .max(1);
    runs.splice(merge_index..=merge_index + 1, [merged]);
    allocations.splice(merge_index..=merge_index + 1, [merged_allocation]);
}

fn split_heaviest_profile_run(runs: &[(usize, usize, u128)], allocations: &mut [usize]) -> bool {
    let Some((run_index, _)) = runs
        .iter()
        .enumerate()
        .filter(|(index, (start, end, _))| allocations[*index] < end - start)
        .max_by(
            |(left_index, (_, _, left_weight)), (right_index, (_, _, right_weight))| {
                let left_score = left_weight.div_ceil(allocations[*left_index] as u128);
                let right_score = right_weight.div_ceil(allocations[*right_index] as u128);
                left_score
                    .cmp(&right_score)
                    .then_with(|| right_weight.cmp(left_weight))
                    .then_with(|| right_index.cmp(left_index))
            },
        )
    else {
        return false;
    };
    allocations[run_index] += 1;
    true
}

fn weighted_greedy_groups(
    values: &[usize],
    weights: &[u128],
    group_count: usize,
) -> Vec<Vec<usize>> {
    debug_assert_eq!(values.len(), weights.len());
    if values.is_empty() {
        return Vec::new();
    }
    let group_count = group_count.max(1).min(values.len());
    let mut groups = (0..group_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut loads = vec![0u128; group_count];
    let mut items = values
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .collect::<Vec<_>>();
    items.sort_by(|&(left_value, left_weight), &(right_value, right_weight)| {
        right_weight
            .cmp(&left_weight)
            .then_with(|| left_value.cmp(&right_value))
    });

    for (value, weight) in items {
        let target = loads
            .iter()
            .enumerate()
            .min_by(|(left_index, left_load), (right_index, right_load)| {
                left_load
                    .cmp(right_load)
                    .then_with(|| left_index.cmp(right_index))
            })
            .map(|(index, _)| index)
            .expect("weighted_greedy_groups has at least one group");
        groups[target].push(value);
        loads[target] = loads[target].saturating_add(weight.max(1));
    }
    for group in &mut groups {
        group.sort_unstable();
    }
    groups.retain(|group| !group.is_empty());
    groups
}

fn group_bounds(group: &[usize]) -> Option<(usize, usize)> {
    Some((*group.first()?, *group.last()?))
}

fn fixed_t_group_ranges(groups: &[Vec<usize>]) -> Vec<(usize, usize)> {
    groups
        .iter()
        .filter_map(|group| group_bounds(group))
        .collect()
}

fn interval_overlap(first: usize, last: usize, other_first: usize, other_last: usize) -> usize {
    let start = first.max(other_first);
    let end = last.min(other_last);
    end.checked_sub(start).map(|delta| delta + 1).unwrap_or(0)
}

fn interval_distance(first: usize, last: usize, other_first: usize, other_last: usize) -> usize {
    if last < other_first {
        other_first - last
    } else {
        first.saturating_sub(other_last)
    }
}

fn product_cache_group_affinity_score(
    group_first: usize,
    group_last: usize,
    persisted: &FixedTBatchPersistedProductCaches,
    index: usize,
) -> (usize, Reverse<usize>, Reverse<usize>) {
    (
        interval_overlap(
            group_first,
            group_last,
            persisted.group_first,
            persisted.group_last,
        ),
        Reverse(interval_distance(
            group_first,
            group_last,
            persisted.group_first,
            persisted.group_last,
        )),
        Reverse(index),
    )
}

fn take_best_persistent_product_cache_for_group(
    stored: &mut [Option<FixedTBatchPersistedProductCaches>],
    group: &[usize],
) -> Option<FixedTBatchPersistedProductCaches> {
    let (group_first, group_last) = group_bounds(group)?;
    let mut best = None;
    for (index, persisted) in stored.iter().enumerate() {
        let Some(persisted) = persisted.as_ref() else {
            continue;
        };
        let score = product_cache_group_affinity_score(group_first, group_last, persisted, index);
        if best
            .map(|(_, best_score)| score > best_score)
            .unwrap_or(true)
        {
            best = Some((index, score));
        }
    }
    let (best_index, _) = best?;
    stored[best_index].take()
}

fn bidegree_counts_at_t(resolution: &Resolution, t: usize) -> BTreeMap<(usize, usize), usize> {
    let mut counts = BTreeMap::new();
    for generator in &resolution.generators {
        if generator.t == t {
            *counts.entry((generator.s, generator.t)).or_default() += 1;
        }
    }
    counts
}

fn basis_index(basis: &[BasisElem]) -> HashMap<(CoeffKey, usize), usize> {
    let mut index = fast_hash_map_with_capacity(basis.len());
    for (i, elem) in basis.iter().enumerate() {
        index.insert(basis_elem_key(elem), i);
    }
    index
}

fn basis_elem_key(elem: &BasisElem) -> (CoeffKey, usize) {
    (elem.coeff_packed, elem.generator)
}

fn xor_terms(terms: &mut Vec<ModuleTerm>, rhs: Vec<ModuleTerm>) {
    let mut counts = fast_hash_map_with_capacity::<ModuleTerm, bool>(terms.len() + rhs.len());
    for term in terms.drain(..).chain(rhs) {
        let entry = counts.entry(term).or_insert(false);
        *entry = !*entry;
    }
    let out = counts
        .into_iter()
        .filter_map(|(term, keep)| keep.then_some(term))
        .collect::<Vec<_>>();
    *terms = out;
}

fn sort_terms(terms: &mut [ModuleTerm]) {
    terms.sort_by(|a, b| {
        a.generator
            .cmp(&b.generator)
            .then_with(|| packed_cmp(a.coeff_packed, b.coeff_packed))
    });
}

struct LinearMap {
    columns: Vec<BitVec>,
    domain_dim: usize,
    target_dim: usize,
}

impl LinearMap {
    fn estimated_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.columns.capacity() * std::mem::size_of::<BitVec>()
            + self
                .columns
                .iter()
                .map(BitVec::storage_bytes)
                .sum::<usize>()
    }
}

struct ProductList(Vec<CoeffKey>);

impl ProductList {
    fn from_vec(products: Vec<CoeffKey>) -> Self {
        Self(products)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn heap_bytes(&self) -> usize {
        self.0.capacity() * std::mem::size_of::<CoeffKey>()
    }

    fn as_slice(&self) -> &[CoeffKey] {
        &self.0
    }
}

struct EmptyProductCache {
    policy: E0EmptyCachePolicy,
    set: FastHashSet<ProductKey>,
    segments: Vec<Box<[ProductKey]>>,
    len: usize,
}

impl Default for EmptyProductCache {
    fn default() -> Self {
        Self {
            policy: E0EmptyCachePolicy::SpeedHash,
            set: FastHashSet::default(),
            segments: Vec::new(),
            len: 0,
        }
    }
}

impl EmptyProductCache {
    fn set_policy(&mut self, policy: E0EmptyCachePolicy) {
        if self.policy == policy {
            return;
        }
        self.clear_and_shrink();
        self.policy = policy;
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        match self.policy {
            E0EmptyCachePolicy::SpeedHash => self.set.len(),
            E0EmptyCachePolicy::CompactSegments => self.len,
        }
    }

    fn segment_count(&self) -> usize {
        match self.policy {
            E0EmptyCachePolicy::SpeedHash => 0,
            E0EmptyCachePolicy::CompactSegments => self.segments.len(),
        }
    }

    fn capacity_items(&self) -> usize {
        match self.policy {
            E0EmptyCachePolicy::SpeedHash => self.set.capacity(),
            E0EmptyCachePolicy::CompactSegments => self.len,
        }
    }

    fn estimated_bytes(&self) -> usize {
        match self.policy {
            E0EmptyCachePolicy::SpeedHash => {
                self.set.capacity()
                    * (std::mem::size_of::<ProductKey>() + std::mem::size_of::<usize>())
            }
            E0EmptyCachePolicy::CompactSegments => {
                self.len * std::mem::size_of::<ProductKey>()
                    + self.segments.capacity() * std::mem::size_of::<Box<[ProductKey]>>()
            }
        }
    }

    fn clear_and_shrink(&mut self) {
        self.set.clear();
        self.set.shrink_to_fit();
        self.segments.clear();
        self.segments.shrink_to_fit();
        self.len = 0;
    }

    fn insert_sorted(&mut self, mut new_empty: Vec<ProductKey>) {
        if new_empty.is_empty() {
            return;
        }
        match self.policy {
            E0EmptyCachePolicy::SpeedHash => {
                self.set.reserve(new_empty.len());
                for key in new_empty {
                    self.set.insert(key);
                }
            }
            E0EmptyCachePolicy::CompactSegments => {
                new_empty.shrink_to_fit();
                self.len += new_empty.len();
                self.segments.push(new_empty.into_boxed_slice());
                if self.segments.len() > E0_EMPTY_SEGMENT_COMPACT_LIMIT {
                    self.compact();
                }
            }
        }
    }

    #[cfg(test)]
    fn filter_missing(&self, missing: &mut Vec<ProductKey>) {
        if self.is_empty() || missing.is_empty() {
            return;
        }
        match self.policy {
            E0EmptyCachePolicy::SpeedHash => {
                missing.retain(|key| !self.set.contains(key));
            }
            E0EmptyCachePolicy::CompactSegments => {
                if self.segments.len() == 1 {
                    filter_missing_against_segment(missing, &self.segments[0]);
                } else {
                    filter_missing_against_segments(missing, &self.segments);
                }
            }
        }
    }

    fn supports_fast_membership_test(&self) -> bool {
        matches!(self.policy, E0EmptyCachePolicy::SpeedHash)
    }

    fn contains(&self, key: &ProductKey) -> bool {
        match self.policy {
            E0EmptyCachePolicy::SpeedHash => self.set.contains(key),
            E0EmptyCachePolicy::CompactSegments => self
                .segments
                .iter()
                .any(|segment| segment.binary_search(key).is_ok()),
        }
    }

    fn compact(&mut self) {
        if self.segments.len() <= 1 {
            return;
        }
        let old_segments = std::mem::take(&mut self.segments);
        let mut merged = Vec::with_capacity(self.len);
        for segment in old_segments {
            merged.extend_from_slice(&segment);
        }
        merged.sort_unstable();
        merged.dedup();
        self.len = merged.len();
        self.segments.push(merged.into_boxed_slice());
        release_allocator_pressure();
    }
}

#[cfg(test)]
fn filter_missing_against_segment(missing: &mut Vec<ProductKey>, segment: &[ProductKey]) {
    let mut index = 0usize;
    missing.retain(|key| {
        while index < segment.len() && segment[index] < *key {
            index += 1;
        }
        index >= segment.len() || segment[index] != *key
    });
}

#[cfg(test)]
fn filter_missing_against_segments(missing: &mut Vec<ProductKey>, segments: &[Box<[ProductKey]>]) {
    let mut indices = vec![0usize; segments.len()];
    let mut heap = BinaryHeap::<(Reverse<ProductKey>, usize)>::with_capacity(segments.len());
    for (segment_index, segment) in segments.iter().enumerate() {
        if let Some(&key) = segment.first() {
            heap.push((Reverse(key), segment_index));
        }
    }

    missing.retain(|key| {
        while let Some(&(Reverse(cached_key), segment_index)) = heap.peek() {
            if cached_key >= *key {
                break;
            }
            heap.pop();
            let segment = &segments[segment_index];
            let mut index = indices[segment_index] + 1;
            while index < segment.len() && segment[index] < *key {
                index += 1;
            }
            indices[segment_index] = index;
            if index < segment.len() {
                heap.push((Reverse(segment[index]), segment_index));
            }
        }
        heap.peek()
            .map(|&(Reverse(cached_key), _)| cached_key != *key)
            .unwrap_or(true)
    });
}

#[derive(Clone)]
struct GeneratorRows {
    offset: usize,
    coeff_index: Arc<HashMap<CoeffKey, usize>>,
}

struct FullDifferentialTermPlan {
    coeff_packed: CoeffKey,
    target_generator: usize,
    rows_offset: usize,
    coeff_index: Arc<HashMap<CoeffKey, usize>>,
    product_width_bound: usize,
    right_width: usize,
}

struct FullBasisLookup {
    generator_rows: Vec<Option<GeneratorRows>>,
    target_dim: usize,
}

impl FullBasisLookup {
    fn estimated_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.generator_rows.capacity() * std::mem::size_of::<Option<GeneratorRows>>()
    }
}

fn bytes_to_mb(bytes: usize) -> f64 {
    bytes as f64 / 1_048_576.0
}

#[cfg(target_os = "macos")]
fn force_release_allocator_pressure() {
    unsafe extern "C" {
        fn malloc_zone_pressure_relief(zone: *mut libc::c_void, goal: libc::size_t)
        -> libc::size_t;
    }
    unsafe {
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
}

#[cfg(target_os = "linux")]
fn force_release_allocator_pressure() {
    unsafe {
        libc::malloc_trim(0);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn force_release_allocator_pressure() {}

fn release_allocator_pressure() {
    if !ALLOCATOR_RELIEF_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    force_release_allocator_pressure();
}

fn process_memory_event_fields() -> String {
    ProcessMemory::now()
        .map(|memory| {
            format!(
                " rss_bytes={} rss_peak_bytes={}",
                memory.resident_bytes, memory.resident_peak_bytes
            )
        })
        .unwrap_or_default()
}

fn subalgebra_log_list(names: &[String]) -> String {
    if names.is_empty() {
        "-".to_string()
    } else {
        names.join("|")
    }
}

fn run_fixed_t_task_with_memory_samples<R, Compute, Progress>(
    progress: &Progress,
    group_index: usize,
    s: usize,
    task_timer: &Instant,
    compute: Compute,
) -> R
where
    Compute: FnOnce() -> R,
    Progress: Fn(&str, String) -> Result<(), String> + Sync,
{
    let Some(interval) = fixed_t_task_memory_sample_interval() else {
        return compute();
    };

    std::thread::scope(|scope| {
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let monitor = scope.spawn(move || {
            loop {
                match stop_rx.recv_timeout(interval) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let _ = progress(
                            "worker_task_memory",
                            format!(
                                "local_group_index={group_index} s={s} wall_s={:.6}{}",
                                task_timer.elapsed().as_secs_f64(),
                                process_memory_event_fields()
                            ),
                        );
                    }
                }
            }
        });
        let result = compute();
        let _ = stop_tx.send(());
        let _ = monitor.join();
        result
    })
}

fn fixed_t_task_memory_sample_interval() -> Option<Duration> {
    static INTERVAL: OnceLock<Option<Duration>> = OnceLock::new();
    *INTERVAL.get_or_init(|| {
        env_duration_secs("EXT_TASK_MEMORY_INTERVAL_SECS")
            .or_else(|| env_duration_secs("EXT_PROCESS_MEMORY_INTERVAL_SECS"))
            .unwrap_or(Some(Duration::from_secs(
                DEFAULT_FIXED_T_TASK_MEMORY_SAMPLE_SECS,
            )))
    })
}

fn env_duration_secs(name: &str) -> Option<Option<Duration>> {
    let raw = std::env::var_os(name)?;
    let seconds = raw.to_string_lossy().parse::<u64>().ok()?;
    Some((seconds > 0).then(|| Duration::from_secs(seconds)))
}

fn estimate_map_table_bytes<K, V, S>(map: &std::collections::HashMap<K, V, S>) -> usize {
    map.capacity() * (std::mem::size_of::<(K, V)>() + std::mem::size_of::<usize>())
}

fn estimate_vec_coeff_map<K, S>(
    map: &std::collections::HashMap<K, Vec<CoeffKey>, S>,
) -> (usize, usize) {
    let terms = map.values().map(Vec::capacity).sum::<usize>();
    let bytes = estimate_map_table_bytes(map) + terms * std::mem::size_of::<CoeffKey>();
    (terms, bytes)
}

fn estimate_product_list_map<K, S>(
    map: &std::collections::HashMap<K, ProductList, S>,
) -> (usize, usize) {
    let terms = map.values().map(ProductList::len).sum::<usize>();
    let bytes =
        estimate_map_table_bytes(map) + map.values().map(ProductList::heap_bytes).sum::<usize>();
    (terms, bytes)
}

fn estimate_e0_profile_bank_bytes<S>(
    bank: &std::collections::HashMap<E0ProfileKey, E0ProfileProductCaches, S>,
) -> usize {
    estimate_map_table_bytes(bank)
        + bank
            .values()
            .map(E0ProfileProductCaches::estimated_bytes)
            .sum::<usize>()
}

fn estimate_computed_e0_bound_bytes(values: &[(ProductKeyWithBound, ProductList)]) -> usize {
    std::mem::size_of_val(values)
        + values
            .iter()
            .map(|(_, product)| product.heap_bytes())
            .sum::<usize>()
}

fn vec_product_key_bound_bytes(values: &Vec<ProductKeyWithBound>) -> usize {
    values.capacity() * std::mem::size_of::<ProductKeyWithBound>()
}

fn shrink_product_key_bound_vec_if_sparse(values: &mut Vec<ProductKeyWithBound>) -> bool {
    let spare_items = values.capacity().saturating_sub(values.len());
    let spare_bytes = spare_items * std::mem::size_of::<ProductKeyWithBound>();
    if spare_bytes >= E0_MISSING_SPARE_SHRINK_BYTES
        && values.capacity() > values.len().saturating_mul(2)
    {
        values.shrink_to_fit();
        true
    } else {
        false
    }
}

fn sort_dedup_product_key_bounds(values: &mut Vec<ProductKeyWithBound>) {
    if values.len() >= PARALLEL_PRODUCT_SORT_MIN_KEYS {
        values.par_sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    } else {
        values.sort_unstable_by_key(|item| (item.0, item.1));
    }
    if values.len() < 2 {
        return;
    }

    let mut write = 1;
    let mut previous_key = (values[0].0, values[0].1);
    for read in 1..values.len() {
        let key = (values[read].0, values[read].1);
        if key == previous_key {
            values[write - 1].2 = values[write - 1].2.max(values[read].2);
        } else {
            previous_key = key;
            values[write] = values[read];
            write += 1;
        }
    }
    values.truncate(write);
}

#[cfg(test)]
fn scoped_full_differential_chunk_bytes_override(
    chunk_bytes: Option<usize>,
) -> FullDifferentialChunkOverrideGuard {
    let previous = FULL_DIFFERENTIAL_CHUNK_BYTES_OVERRIDE.with(|override_bytes| {
        let previous = override_bytes.get();
        override_bytes.set(chunk_bytes);
        previous
    });
    FullDifferentialChunkOverrideGuard { previous }
}

#[cfg(test)]
fn scoped_full_differential_output_chunk_bytes_override(
    chunk_bytes: Option<usize>,
) -> FullDifferentialOutputChunkOverrideGuard {
    let previous = FULL_DIFFERENTIAL_OUTPUT_CHUNK_BYTES_OVERRIDE.with(|override_bytes| {
        let previous = override_bytes.get();
        override_bytes.set(chunk_bytes);
        previous
    });
    FullDifferentialOutputChunkOverrideGuard { previous }
}

fn configured_full_differential_chunk_bytes() -> usize {
    static CHUNK_BYTES: OnceLock<usize> = OnceLock::new();
    *CHUNK_BYTES.get_or_init(|| {
        let Some(raw) = std::env::var_os("EXT_FULL_DIFFERENTIAL_CHUNK_MB") else {
            return DEFAULT_FULL_DIFFERENTIAL_CHUNK_BYTES;
        };
        let raw = raw.to_string_lossy();
        match raw.trim().parse::<usize>() {
            Ok(0) => DEFAULT_FULL_DIFFERENTIAL_CHUNK_BYTES,
            Ok(mib) => mib.saturating_mul(1024 * 1024),
            Err(_) => {
                eprintln!(
                    "warning: ignoring invalid EXT_FULL_DIFFERENTIAL_CHUNK_MB={raw}; using {} MiB",
                    DEFAULT_FULL_DIFFERENTIAL_CHUNK_BYTES / (1024 * 1024),
                );
                DEFAULT_FULL_DIFFERENTIAL_CHUNK_BYTES
            }
        }
    })
}

fn full_differential_chunk_bytes() -> usize {
    FULL_DIFFERENTIAL_CHUNK_BYTES_OVERRIDE
        .with(|override_bytes| override_bytes.get())
        .unwrap_or_else(configured_full_differential_chunk_bytes)
}

fn full_differential_output_chunk_bytes() -> Option<usize> {
    FULL_DIFFERENTIAL_OUTPUT_CHUNK_BYTES_OVERRIDE.with(|override_bytes| override_bytes.get())
}

fn full_differential_output_batch_len(target_dim: usize, vector_count: usize) -> usize {
    if vector_count == 0 {
        return 1;
    }
    let Some(chunk_bytes) = full_differential_output_chunk_bytes() else {
        return vector_count;
    };
    let bytes_per_output = BitVec::storage_bytes_for_len(target_dim);
    chunk_bytes
        .checked_div(bytes_per_output)
        .unwrap_or(vector_count)
        .max(1)
        .min(vector_count)
}

fn e0_scan_empty_skip_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("EXT_E0_SCAN_EMPTY_SKIP")
            .map(|raw| raw.trim() != "0")
            .unwrap_or(true)
    })
}

fn bitvec_slice_bytes(vectors: &[BitVec]) -> usize {
    std::mem::size_of_val(vectors) + vectors.iter().map(BitVec::storage_bytes).sum::<usize>()
}

fn estimate_arc_vec_cache<K, T, S>(map: &std::collections::HashMap<K, Arc<Vec<T>>, S>) -> usize {
    estimate_map_table_bytes(map)
        + map
            .values()
            .map(|values| values.capacity() * std::mem::size_of::<T>())
            .sum::<usize>()
}

fn estimate_arc_map_cache<K, IK, IV, S1, S2>(
    map: &std::collections::HashMap<K, Arc<std::collections::HashMap<IK, IV, S2>>, S1>,
) -> usize {
    estimate_map_table_bytes(map)
        + map
            .values()
            .map(|inner| estimate_map_table_bytes(inner.as_ref()))
            .sum::<usize>()
}

fn estimate_row_decomp_cache(
    cache: &HashMap<(u32, usize), RowDecompOptions>,
) -> (usize, usize, usize) {
    let rows = cache
        .values()
        .map(RowDecompOptions::row_count)
        .sum::<usize>();
    let options = cache
        .values()
        .map(RowDecompOptions::storage_capacity)
        .sum::<usize>();
    let bytes = estimate_map_table_bytes(cache) + options * std::mem::size_of::<u32>();
    (rows, options, bytes)
}

fn translate_to_zero(vector: &BitVec, translation: &SignatureTranslation) -> BitVec {
    debug_assert_eq!(vector.len(), translation.current_len);
    if translation.identity {
        return vector.clone();
    }
    let mut out = BitVec::new(translation.zero_len);
    for i in vector.ones() {
        out.toggle(translation.to_zero[i]);
    }
    out
}

fn translate_from_zero(vector: &BitVec, translation: &SignatureTranslation) -> BitVec {
    debug_assert_eq!(vector.len(), translation.zero_len);
    if translation.identity {
        return vector.clone();
    }
    let mut out = BitVec::new(translation.current_len);
    for i in vector.ones() {
        out.toggle(translation.from_zero[i]);
    }
    out
}

fn xor_full_columns_into_outputs(
    columns: &[BitVec],
    groups: &[(usize, usize, usize)],
    uses: &[(usize, usize)],
    outputs: &mut [BitVec],
) {
    debug_assert_eq!(columns.len(), groups.len());
    for (column, &(_, start, end)) in columns.iter().zip(groups) {
        for &(_, vector_index) in &uses[start..end] {
            outputs[vector_index].xor_assign(column);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_product_cache_filters_across_segments() {
        let mut cache = EmptyProductCache::default();
        cache.insert_sorted(vec![(1, 1), (2, 4), (5, 8)]);
        cache.insert_sorted(vec![(1, 2), (3, 5), (8, 13)]);
        cache.insert_sorted(vec![(2, 4), (6, 9), (10, 11)]);

        assert!(cache.supports_fast_membership_test());
        assert!(cache.contains(&(2, 4)));
        assert!(!cache.contains(&(4, 4)));

        let mut missing = vec![
            (1, 1),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 4),
            (8, 13),
            (9, 1),
            (10, 11),
        ];
        cache.filter_missing(&mut missing);

        assert_eq!(missing, vec![(1, 3), (4, 4), (9, 1)]);
    }

    #[test]
    fn fixed_t_layers_cover_the_full_adams_triangle() {
        assert_eq!(fixed_t_task_s_limit(0), None);
        for t in 1..=64 {
            assert_eq!(fixed_t_task_s_limit(t), Some(t - 1));
            assert_eq!(
                ComputeCursor::after_step(t, t - 1),
                ComputeCursor {
                    next_t: t + 1,
                    next_s: 0,
                }
            );
        }
    }

    #[test]
    fn milnor_basis_grows_only_when_each_t_layer_is_reached() {
        let mut resolution = Resolution::new(140);
        assert_eq!(resolution.algebra_basis.len(), 1);

        let mut completed_layers = Vec::new();
        resolution
            .compute_from_cursor_with_progress(
                8,
                ComputeMode::Naive,
                ComputeCursor::start(),
                |resolution, progress| {
                    if progress.next_s == 0 {
                        completed_layers.push((progress.t, resolution.algebra_basis.len()));
                    }
                    Ok(())
                },
            )
            .unwrap();

        assert_eq!(
            completed_layers,
            (1..=8).map(|t| (t, t + 1)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn first_generators_match_low_degree_expectations() {
        let mut res = Resolution::new(8);
        res.compute(8, ComputeMode::Naive).unwrap();
        let h1 = res
            .generators
            .iter()
            .filter(|g| g.s == 1 && matches!(g.t, 1 | 2 | 4 | 8))
            .count();
        assert_eq!(h1, 4);
        assert!(
            res.generators
                .iter()
                .any(|g| g.s == 2 && g.t == 2 && g.differential.len() == 1)
        );
    }

    #[test]
    fn start_cursor_preserves_initial_sphere_generator_as_seed() {
        let mut res = Resolution::new(0);
        let cursor = res
            .compute_from_cursor(0, ComputeMode::Naive, ComputeCursor::start())
            .unwrap();
        assert_eq!(
            cursor,
            ComputeCursor {
                next_t: 1,
                next_s: 0
            }
        );
        assert_eq!(res.generators.len(), 1);
        assert_eq!((res.generators[0].s, res.generators[0].t), (0, 0));
        assert!(res.generators[0].differential.is_empty());
    }

    #[test]
    fn subalgebra_selection_uses_task_s_not_output_q() {
        let mut res = Resolution::new(2);
        res.ensure_algebra_basis_through(2);
        let a0 = Subalgebra::a(0).unwrap();
        assert!(
            a0.selection_condition_applies(0, 2),
            "A0 should be justified for task_s=0 at t=2"
        );
        assert!(
            !a0.selection_condition_applies(1, 2),
            "using output q=task_s+1 would incorrectly reject/shift this boundary case"
        );
        let candidates = vec![a0.clone()];
        let chosen = res.choose_subalgebra(0, 2, &candidates, false).unwrap();
        assert_eq!(chosen.name(), "A0");
    }

    #[test]
    fn algorithm2_a0_agrees_with_naive_in_small_range() {
        let max_t = 10;
        let mut naive = Resolution::new(max_t);
        naive.compute(max_t, ComputeMode::Naive).unwrap();

        let mut accelerated = Resolution::new(max_t);
        accelerated
            .compute(
                max_t,
                ComputeMode::Accelerated {
                    subalgebra: Subalgebra::a(0).unwrap(),
                    strict: false,
                    force: false,
                },
            )
            .unwrap();

        let naive_degrees = naive
            .generators
            .iter()
            .map(|g| (g.s, g.t))
            .collect::<Vec<_>>();
        let algorithm2_degrees = accelerated
            .generators
            .iter()
            .map(|g| (g.s, g.t))
            .collect::<Vec<_>>();
        assert_eq!(algorithm2_degrees, naive_degrees);
    }

    #[test]
    fn auto_mode_agrees_with_naive_in_small_range() {
        let max_t = 10;
        let mut naive = Resolution::new(max_t);
        naive.compute(max_t, ComputeMode::Naive).unwrap();

        let mut auto = Resolution::new(max_t);
        auto.compute(
            max_t,
            ComputeMode::Auto {
                subalgebras: vec![
                    Subalgebra::a(2).unwrap(),
                    Subalgebra::a(1).unwrap(),
                    Subalgebra::a(0).unwrap(),
                ],
                strict: false,
                force: false,
            },
        )
        .unwrap();

        let naive_degrees = naive
            .generators
            .iter()
            .map(|g| (g.s, g.t))
            .collect::<Vec<_>>();
        let auto_degrees = auto
            .generators
            .iter()
            .map(|g| (g.s, g.t))
            .collect::<Vec<_>>();
        assert_eq!(auto_degrees, naive_degrees);
    }

    #[test]
    fn chunked_signature_lift_solving_matches_auto_small_range() {
        let max_t = 10;
        let mut naive = Resolution::new(max_t);
        naive.compute(max_t, ComputeMode::Naive).unwrap();

        let mut auto = Resolution::new(max_t);
        auto.compute(
            max_t,
            ComputeMode::Auto {
                subalgebras: vec![
                    Subalgebra::a(2).unwrap(),
                    Subalgebra::a(1).unwrap(),
                    Subalgebra::a(0).unwrap(),
                ],
                strict: false,
                force: false,
            },
        )
        .unwrap();

        INITIAL_FULL_DIFFERENTIAL_BATCH_TEST_HITS.store(0, Ordering::Relaxed);
        SIGNATURE_LIFT_BATCH_TEST_HITS.store(0, Ordering::Relaxed);
        let guard = scoped_full_differential_output_chunk_bytes_override(Some(1));
        let mut chunked = Resolution::new(max_t);
        chunked
            .compute(
                max_t,
                ComputeMode::Auto {
                    subalgebras: vec![
                        Subalgebra::a(2).unwrap(),
                        Subalgebra::a(1).unwrap(),
                        Subalgebra::a(0).unwrap(),
                    ],
                    strict: false,
                    force: false,
                },
            )
            .unwrap();
        drop(guard);

        assert!(
            INITIAL_FULL_DIFFERENTIAL_BATCH_TEST_HITS.load(Ordering::Relaxed) > 0,
            "chunked initial full differential path was not exercised"
        );
        assert!(
            SIGNATURE_LIFT_BATCH_TEST_HITS.load(Ordering::Relaxed) > 0,
            "chunked signature lift solve path was not exercised"
        );
        assert_eq!(resolution_signature(&chunked), resolution_signature(&auto));
        assert_eq!(generator_counts(&chunked), generator_counts(&naive));
        assert_resolution_valid_through(&mut chunked, max_t);
        assert_exact_through(&mut chunked, max_t);
    }

    #[test]
    fn resume_from_completed_t_layer_matches_fresh_auto() {
        let max_t = 10;
        let mut partial = Resolution::new(max_t);
        partial
            .compute(
                5,
                ComputeMode::Auto {
                    subalgebras: vec![
                        Subalgebra::a(2).unwrap(),
                        Subalgebra::a(1).unwrap(),
                        Subalgebra::a(0).unwrap(),
                    ],
                    strict: false,
                    force: false,
                },
            )
            .unwrap();
        let snapshot = partial.snapshot();
        let mut resumed = Resolution::from_snapshot(max_t, snapshot).unwrap();
        resumed
            .compute_from_cursor(
                max_t,
                ComputeMode::Auto {
                    subalgebras: vec![
                        Subalgebra::a(2).unwrap(),
                        Subalgebra::a(1).unwrap(),
                        Subalgebra::a(0).unwrap(),
                    ],
                    strict: false,
                    force: false,
                },
                ComputeCursor {
                    next_t: 6,
                    next_s: 0,
                },
            )
            .unwrap();

        let mut fresh = Resolution::new(max_t);
        fresh
            .compute(
                max_t,
                ComputeMode::Auto {
                    subalgebras: vec![
                        Subalgebra::a(2).unwrap(),
                        Subalgebra::a(1).unwrap(),
                        Subalgebra::a(0).unwrap(),
                    ],
                    strict: false,
                    force: false,
                },
            )
            .unwrap();

        assert_eq!(resolution_signature(&resumed), resolution_signature(&fresh));
    }

    #[test]
    fn resume_from_mid_layer_matches_fresh_auto() {
        let max_t = 10;
        let mut partial = Resolution::new(max_t);
        let mut cursor = ComputeCursor::start();
        let mut steps = 0;
        let stopped = partial.compute_from_cursor_with_progress(
            max_t,
            ComputeMode::Auto {
                subalgebras: vec![
                    Subalgebra::a(2).unwrap(),
                    Subalgebra::a(1).unwrap(),
                    Subalgebra::a(0).unwrap(),
                ],
                strict: false,
                force: false,
            },
            ComputeCursor::start(),
            |_, progress| {
                cursor = ComputeCursor {
                    next_t: progress.next_t,
                    next_s: progress.next_s,
                };
                steps += 1;
                if steps == 17 {
                    Err("intentional stop".to_string())
                } else {
                    Ok(())
                }
            },
        );
        assert!(stopped.is_err());
        assert_eq!(cursor.next_t, 6);
        assert_eq!(cursor.next_s, 2);

        let snapshot = partial.snapshot();
        let mut resumed = Resolution::from_snapshot(max_t, snapshot).unwrap();
        resumed
            .compute_from_cursor(
                max_t,
                ComputeMode::Auto {
                    subalgebras: vec![
                        Subalgebra::a(2).unwrap(),
                        Subalgebra::a(1).unwrap(),
                        Subalgebra::a(0).unwrap(),
                    ],
                    strict: false,
                    force: false,
                },
                cursor,
            )
            .unwrap();

        let mut fresh = Resolution::new(max_t);
        fresh
            .compute(
                max_t,
                ComputeMode::Auto {
                    subalgebras: vec![
                        Subalgebra::a(2).unwrap(),
                        Subalgebra::a(1).unwrap(),
                        Subalgebra::a(0).unwrap(),
                    ],
                    strict: false,
                    force: false,
                },
            )
            .unwrap();

        assert_eq!(resolution_signature(&resumed), resolution_signature(&fresh));
    }

    #[test]
    fn cache_invalidation_keeps_degrees_below_new_generator() {
        let mut resolution = Resolution::new(20);
        let subalgebra = Subalgebra::a(0).unwrap();
        let low_key = resolution.signature_basis_key(1, 5, &subalgebra, 0);
        let high_key = resolution.signature_basis_key(1, 12, &subalgebra, 0);
        let low_target_key = resolution.signature_basis_key(2, 5, &subalgebra, 0);
        let high_target_key = resolution.signature_basis_key(2, 12, &subalgebra, 0);

        resolution.basis_cache.insert((1, 5), Arc::new(Vec::new()));
        resolution.basis_cache.insert((1, 12), Arc::new(Vec::new()));
        resolution
            .basis_index_cache
            .insert((1, 5), Arc::new(HashMap::default()));
        resolution
            .basis_index_cache
            .insert((1, 12), Arc::new(HashMap::default()));
        resolution.full_basis_lookup_cache.insert(
            (1, 5),
            Arc::new(FullBasisLookup {
                generator_rows: Vec::new(),
                target_dim: 0,
            }),
        );
        resolution.full_basis_lookup_cache.insert(
            (1, 12),
            Arc::new(FullBasisLookup {
                generator_rows: Vec::new(),
                target_dim: 0,
            }),
        );
        resolution
            .basis_signature_cache
            .insert(low_key, Arc::new(Vec::new()));
        resolution
            .basis_signature_cache
            .insert(high_key, Arc::new(Vec::new()));
        resolution.signature_matrix_cache.insert(
            low_target_key,
            Arc::new(LinearMap {
                columns: Vec::new(),
                domain_dim: 0,
                target_dim: 0,
            }),
        );
        resolution.signature_matrix_cache.insert(
            high_target_key,
            Arc::new(LinearMap {
                columns: Vec::new(),
                domain_dim: 0,
                target_dim: 0,
            }),
        );
        resolution
            .signature_solver_cache
            .insert(low_target_key, Arc::new(LinearSolver::new(&[], 0)));
        resolution
            .signature_solver_cache
            .insert(high_target_key, Arc::new(LinearSolver::new(&[], 0)));

        resolution.add_generator(1, 10, Vec::new());

        assert!(resolution.basis_cache.contains_key(&(1, 5)));
        assert!(!resolution.basis_cache.contains_key(&(1, 12)));
        assert!(resolution.basis_index_cache.contains_key(&(1, 5)));
        assert!(!resolution.basis_index_cache.contains_key(&(1, 12)));
        assert!(resolution.full_basis_lookup_cache.contains_key(&(1, 5)));
        assert!(!resolution.full_basis_lookup_cache.contains_key(&(1, 12)));
        assert!(resolution.basis_signature_cache.contains_key(&low_key));
        assert!(!resolution.basis_signature_cache.contains_key(&high_key));
        assert!(
            resolution
                .signature_matrix_cache
                .contains_key(&low_target_key)
        );
        assert!(
            !resolution
                .signature_matrix_cache
                .contains_key(&high_target_key)
        );
        assert!(
            resolution
                .signature_solver_cache
                .contains_key(&low_target_key)
        );
        assert!(
            !resolution
                .signature_solver_cache
                .contains_key(&high_target_key)
        );
    }

    #[test]
    fn fixed_t_worker_cache_delta_excludes_inherited_entries() {
        let mut baseline = Resolution::new(20);
        let subalgebra = Subalgebra::a(0).unwrap();
        let inherited_key = baseline.signature_basis_key(1, 5, &subalgebra, 0);
        let new_key = baseline.signature_basis_key(2, 6, &subalgebra, 0);

        baseline.basis_cache.insert((1, 5), Arc::new(Vec::new()));
        baseline
            .basis_signature_cache
            .insert(inherited_key, Arc::new(Vec::new()));
        baseline.signature_matrix_cache.insert(
            inherited_key,
            Arc::new(LinearMap {
                columns: Vec::new(),
                domain_dim: 0,
                target_dim: 0,
            }),
        );
        baseline
            .signature_solver_cache
            .insert(inherited_key, Arc::new(LinearSolver::new(&[], 0)));

        let mut worker = baseline.clone_with_shared_worker_caches();
        worker.basis_cache.insert((2, 6), Arc::new(Vec::new()));
        worker
            .basis_signature_cache
            .insert(new_key, Arc::new(Vec::new()));
        worker.signature_matrix_cache.insert(
            new_key,
            Arc::new(LinearMap {
                columns: Vec::new(),
                domain_dim: 0,
                target_dim: 0,
            }),
        );
        worker
            .signature_solver_cache
            .insert(new_key, Arc::new(LinearSolver::new(&[], 0)));

        let delta = worker.take_fixed_t_worker_caches_new_since(&baseline);

        assert!(!delta.basis_cache.contains_key(&(1, 5)));
        assert!(delta.basis_cache.contains_key(&(2, 6)));
        assert!(!delta.basis_signature_cache.contains_key(&inherited_key));
        assert!(delta.basis_signature_cache.contains_key(&new_key));
        assert!(!delta.signature_matrix_cache.contains_key(&inherited_key));
        assert!(delta.signature_matrix_cache.contains_key(&new_key));
        assert!(!delta.signature_solver_cache.contains_key(&inherited_key));
        assert!(delta.signature_solver_cache.contains_key(&new_key));
    }

    #[test]
    fn fixed_t_worker_cache_merge_drops_entries_stale_after_commit() {
        let mut resolution = Resolution::new(20);
        let subalgebra = Subalgebra::a(0).unwrap();
        let stale_basis_key = resolution.signature_basis_key(3, 10, &subalgebra, 0);
        let stale_domain_key = resolution.signature_basis_key(3, 10, &subalgebra, 1);
        let stale_target_key = resolution.signature_basis_key(4, 10, &subalgebra, 1);
        let kept_low_key = resolution.signature_basis_key(3, 9, &subalgebra, 0);
        let kept_other_key = resolution.signature_basis_key(2, 10, &subalgebra, 0);

        let mut caches = FixedTBatchWorkerCaches::default();
        caches.basis_cache.insert((3, 10), Arc::new(Vec::new()));
        caches.basis_cache.insert((3, 9), Arc::new(Vec::new()));
        caches.basis_cache.insert((2, 10), Arc::new(Vec::new()));
        caches
            .basis_signature_cache
            .insert(stale_basis_key, Arc::new(Vec::new()));
        caches
            .basis_signature_cache
            .insert(kept_low_key, Arc::new(Vec::new()));
        caches
            .basis_signature_cache
            .insert(kept_other_key, Arc::new(Vec::new()));
        for key in [
            stale_domain_key,
            stale_target_key,
            kept_low_key,
            kept_other_key,
        ] {
            caches.signature_matrix_cache.insert(
                key,
                Arc::new(LinearMap {
                    columns: Vec::new(),
                    domain_dim: 0,
                    target_dim: 0,
                }),
            );
            caches
                .signature_solver_cache
                .insert(key, Arc::new(LinearSolver::new(&[], 0)));
        }

        filter_fixed_t_worker_caches_for_commit(&mut caches, 10, &[3]);
        resolution.merge_fixed_t_worker_caches(caches);

        assert!(!resolution.basis_cache.contains_key(&(3, 10)));
        assert!(resolution.basis_cache.contains_key(&(3, 9)));
        assert!(resolution.basis_cache.contains_key(&(2, 10)));
        assert!(
            !resolution
                .basis_signature_cache
                .contains_key(&stale_basis_key)
        );
        assert!(resolution.basis_signature_cache.contains_key(&kept_low_key));
        assert!(
            resolution
                .basis_signature_cache
                .contains_key(&kept_other_key)
        );
        assert!(
            !resolution
                .signature_matrix_cache
                .contains_key(&stale_domain_key)
        );
        assert!(
            !resolution
                .signature_matrix_cache
                .contains_key(&stale_target_key)
        );
        assert!(
            resolution
                .signature_matrix_cache
                .contains_key(&kept_low_key)
        );
        assert!(
            resolution
                .signature_matrix_cache
                .contains_key(&kept_other_key)
        );
        assert!(
            !resolution
                .signature_solver_cache
                .contains_key(&stale_domain_key)
        );
        assert!(
            !resolution
                .signature_solver_cache
                .contains_key(&stale_target_key)
        );
        assert!(
            resolution
                .signature_solver_cache
                .contains_key(&kept_low_key)
        );
        assert!(
            resolution
                .signature_solver_cache
                .contains_key(&kept_other_key)
        );
    }

    #[test]
    fn fixed_t_worker_clone_shares_frozen_read_only_data() {
        let mut resolution = Resolution::new(8);
        resolution.set_cache_policy(CachePolicy {
            e0_empty_cache: E0EmptyCachePolicy::CompactSegments,
            ..CachePolicy::default()
        });
        resolution.add_generator(
            1,
            2,
            vec![ModuleTerm {
                coeff_packed: Milnor::new(vec![1]).packed().unwrap(),
                generator: 0,
            }],
        );
        let cloned = resolution.clone_with_shared_worker_caches();

        assert!(Arc::ptr_eq(
            &resolution.algebra_basis,
            &cloned.algebra_basis
        ));
        assert!(Arc::ptr_eq(
            &resolution.generators[1].differential,
            &cloned.generators[1].differential
        ));
        assert_eq!(
            cloned.cache_policy.e0_empty_cache,
            E0EmptyCachePolicy::CompactSegments
        );
        assert_eq!(
            cloned.mult_e0_empty_cache.policy,
            E0EmptyCachePolicy::CompactSegments
        );
    }

    #[test]
    fn weighted_contiguous_groups_balance_prefix_heavy_work() {
        let values = (0..12).collect::<Vec<_>>();
        let weights = vec![10, 9, 8, 7, 1, 1, 1, 1, 1, 1, 1, 1];
        let groups = weighted_contiguous_groups(&values, &weights, 4);
        let weighted_max = groups
            .iter()
            .map(|group| group.iter().map(|&s| weights[s]).sum::<u128>())
            .max()
            .unwrap();
        let contiguous_max = contiguous_groups(&values, 4)
            .iter()
            .map(|group| group.iter().map(|&s| weights[s]).sum::<u128>())
            .max()
            .unwrap();
        assert!(weighted_max < contiguous_max);
        assert!(groups[0].last().copied().unwrap() < 3);
    }

    #[test]
    fn weighted_contiguous_groups_keep_ranges_contiguous() {
        let values = (3..18).collect::<Vec<_>>();
        let weights = values
            .iter()
            .map(|value| if *value < 8 { 10 } else { 1 })
            .collect::<Vec<_>>();
        let groups = weighted_contiguous_groups(&values, &weights, 5);
        for group in groups {
            for window in group.windows(2) {
                assert_eq!(window[0] + 1, window[1]);
            }
        }
    }

    #[test]
    fn minimax_contiguous_groups_use_exact_group_count() {
        let values = (0..5).collect::<Vec<_>>();
        let weights = vec![1, 1, 1, 1, 5];
        let weighted = weighted_contiguous_groups(&values, &weights, 2);
        assert_eq!(weighted, vec![vec![0, 1, 2, 3], vec![4]]);

        let minimax = minimax_contiguous_groups(&values, &weights, 2);
        assert_eq!(minimax, vec![vec![0, 1, 2, 3], vec![4]]);
        let max_load = minimax
            .iter()
            .map(|group| group.iter().map(|&s| weights[s]).sum::<u128>())
            .max()
            .unwrap();
        assert_eq!(max_load, 5);
    }

    #[test]
    fn fixed_t_local_s_groups_split_each_s_when_workers_are_available() {
        let basis_dims = vec![1, 1, 1, 1, 1, 1, 10, 1, 100, 1];
        let groups = fixed_t_local_s_groups(&[7, 8], &basis_dims, 8);
        assert_eq!(groups, vec![vec![7], vec![8]]);
    }

    #[test]
    fn fixed_t_local_s_groups_use_exact_nonempty_group_count() {
        let values = (0..5).collect::<Vec<_>>();
        let basis_dims = vec![1, 1, 1, 1, 5, 1];
        let groups = fixed_t_local_s_groups(&values, &basis_dims, 3);
        assert_eq!(groups.len(), 3);
        assert!(groups.iter().all(|group| !group.is_empty()));
        assert_eq!(groups.iter().flatten().copied().collect::<Vec<_>>(), values);
        for group in groups {
            for window in group.windows(2) {
                assert_eq!(window[0] + 1, window[1]);
            }
        }
    }

    #[test]
    fn scheduler_task_weight_uses_signature_reduced_dimensions_for_algorithm2() {
        let mut resolution = Resolution::new(24);
        let t = 20;
        let s = 0;
        resolution.ensure_algebra_basis_through(t);
        let basis_dims = (0..=s + 1)
            .map(|index| resolution.basis_dimension_uncached(index, t))
            .collect::<Vec<_>>();
        let raw_weight = estimate_fixed_t_task_weight_from_dims(s, &basis_dims);
        let inner = ComputeMode::Accelerated {
            subalgebra: Subalgebra::a(0).unwrap(),
            strict: false,
            force: true,
        };
        let reduced_weight = resolution.fixed_t_scheduler_task_weight(t, s, &basis_dims, &inner);

        assert!(
            reduced_weight < raw_weight,
            "scheduler should use signature-reduced dimensions for Algorithm 2 tasks: raw={raw_weight} reduced={reduced_weight}"
        );
    }

    #[test]
    fn selector_uses_finite_profile_priority_when_costs_tie() {
        let mut resolution = Resolution::new(170);
        let subalgebras = vec![
            Subalgebra::a(3).unwrap(),
            Subalgebra::b3321().unwrap(),
            Subalgebra::b3221().unwrap(),
            Subalgebra::b3211().unwrap(),
            Subalgebra::a(2).unwrap(),
            Subalgebra::a(1).unwrap(),
            Subalgebra::a(0).unwrap(),
        ];
        let mode = ComputeMode::Auto {
            subalgebras,
            strict: false,
            force: false,
        };

        let b3321 = resolution
            .selected_algorithm2_subalgebra_for_mode(6, 160, &mode)
            .unwrap();
        assert_eq!(b3321.name(), "B3321");

        let b3221 = resolution
            .selected_algorithm2_subalgebra_for_mode(7, 160, &mode)
            .unwrap();
        assert_eq!(b3221.name(), "B3221");

        let b3211 = resolution
            .selected_algorithm2_subalgebra_for_mode(8, 160, &mode)
            .unwrap();
        assert_eq!(b3211.name(), "B3211");
    }

    #[test]
    fn selector_prefers_f_over_matching_fp_before_reduced_dimensions() {
        let f1 = Subalgebra::f(1, 80).unwrap();
        let fp1 = Subalgebra::fprime(1, 80).unwrap();
        let f2 = Subalgebra::f(2, 80).unwrap();
        let fp2 = Subalgebra::fprime(2, 80).unwrap();
        let f3 = Subalgebra::f(3, 80).unwrap();
        let fp3 = Subalgebra::fprime(3, 80).unwrap();

        assert!(candidate_subalgebra_is_better(
            &f1,
            (100, f1.selection_priority()),
            &fp1,
            (1, fp1.selection_priority())
        ));
        assert!(!candidate_subalgebra_is_better(
            &fp1,
            (1, fp1.selection_priority()),
            &f1,
            (100, f1.selection_priority())
        ));
        assert!(candidate_subalgebra_is_better(
            &f2,
            (100, f2.selection_priority()),
            &fp2,
            (1, fp2.selection_priority())
        ));
        assert!(!candidate_subalgebra_is_better(
            &fp2,
            (1, fp2.selection_priority()),
            &f2,
            (100, f2.selection_priority())
        ));
        assert!(!candidate_subalgebra_is_better(
            &f3,
            (100, f3.selection_priority()),
            &fp3,
            (1, fp3.selection_priority())
        ));
    }

    #[test]
    fn selector_disables_fp3_before_force_or_priority() {
        let mut resolution = Resolution::new(80);
        let fp3 = Subalgebra::fprime(3, 80).unwrap();
        let a0 = Subalgebra::a(0).unwrap();
        let candidates = vec![fp3.clone(), a0.clone()];

        let chosen = resolution
            .choose_subalgebra(40, 40, &candidates, true)
            .map(Subalgebra::name);
        assert_eq!(chosen.as_deref(), Some("A0"));

        let mode = ComputeMode::Auto {
            subalgebras: candidates,
            strict: false,
            force: true,
        };
        let names = resolution.candidate_subalgebra_names_for_mode(40, 40, &mode);
        assert!(!names.iter().any(|name| name == "Fp3"));

        let explicit_fp3 = ComputeMode::Accelerated {
            subalgebra: fp3,
            strict: false,
            force: true,
        };
        assert!(resolution.compute_step(40, 40, &explicit_fp3).is_err());
    }

    fn manual_signature_build_chain_cost(
        resolution: &mut Resolution,
        p: usize,
        t: usize,
        subalgebra: &Subalgebra,
    ) -> u128 {
        let generator_ids = resolution.gens_by_s.get(p).cloned().unwrap_or_default();
        generator_ids
            .into_iter()
            .filter_map(|generator_id| {
                let generator_t = resolution.generators[generator_id].t;
                if generator_t > t {
                    return None;
                }
                let differential_len = resolution.generators[generator_id].differential.len();
                let q_b = resolution
                    .coeff_signature_basis_cached(t - generator_t, subalgebra, 0)
                    .len();
                Some((q_b as u128).saturating_mul(differential_len as u128))
            })
            .sum::<u128>()
    }

    #[test]
    fn build_cost_weight_counts_signature_coefficients_times_differential_terms() {
        let mut resolution = Resolution::new(8);
        let xi1 = Milnor::new(vec![1]).packed().unwrap();
        let xi2 = Milnor::new(vec![2]).packed().unwrap();
        let g1 = resolution.add_generator(
            1,
            1,
            vec![ModuleTerm {
                coeff_packed: xi1,
                generator: 0,
            }],
        );
        resolution.add_generator(
            1,
            2,
            vec![
                ModuleTerm {
                    coeff_packed: xi1,
                    generator: 0,
                },
                ModuleTerm {
                    coeff_packed: xi2,
                    generator: 0,
                },
            ],
        );
        resolution.add_generator(
            2,
            3,
            vec![
                ModuleTerm {
                    coeff_packed: xi1,
                    generator: g1,
                },
                ModuleTerm {
                    coeff_packed: xi2,
                    generator: g1,
                },
            ],
        );
        let subalgebra = Subalgebra::a(0).unwrap();
        let t = 5;
        resolution.ensure_algebra_basis_through(t);

        let expected_chain_1 =
            manual_signature_build_chain_cost(&mut resolution, 1, t, &subalgebra);
        let expected_chain_2 =
            manual_signature_build_chain_cost(&mut resolution, 2, t, &subalgebra);
        assert_eq!(
            resolution.estimate_fixed_t_chain_group_build_cost(1, t, &subalgebra),
            expected_chain_1
        );
        assert_eq!(
            resolution.estimate_fixed_t_chain_group_build_cost(2, t, &subalgebra),
            expected_chain_2
        );
        assert_eq!(
            resolution.estimate_fixed_t_task_build_cost_from_signature_pairs(1, t, &subalgebra),
            expected_chain_1.saturating_add(expected_chain_2).max(1)
        );
    }

    #[test]
    fn build_cost_weight_keeps_adams_tail_estimated() {
        let mut resolution = Resolution::new(8);
        let active_s = vec![0, 1, 2, 3, 4, 5];
        let basis_dims = vec![100; 8];
        let inner = ComputeMode::Accelerated {
            subalgebra: Subalgebra::a(0).unwrap(),
            strict: false,
            force: true,
        };
        let weights = resolution.fixed_t_scheduler_build_cost_task_weights(
            12,
            &active_s,
            &basis_dims,
            &inner,
        );
        assert!(
            weights[..=4].iter().any(|&weight| weight > 0),
            "non-tail weights should still be estimated: {weights:?}"
        );
        assert!(
            weights[5] > 0,
            "build-cost scheduling should estimate Adams tail work instead of collapsing it to zero: {weights:?}"
        );
    }

    #[test]
    fn build_cost_subalgebra_multipliers_match_calibration() {
        let max_degree = 32;
        let cases = [
            ("A3", (15, 1)),
            ("B3321", (3, 1)),
            ("B3221", (5, 1)),
            ("B3211", (3, 1)),
            ("A2", (2, 1)),
            ("A1", (1, 1)),
            ("A0", (1, 1)),
            ("Fp3", (1, 1)),
            ("F2", (1, 1)),
            ("Fp2", (1, 1)),
            ("F1", (1, 1)),
            ("Fp1", (1, 1)),
        ];

        for (name, expected) in cases {
            let subalgebra = Subalgebra::parse(name, max_degree).unwrap();
            assert_eq!(
                fixed_t_subalgebra_build_cost_multiplier(&subalgebra),
                expected,
                "unexpected build-cost multiplier for {name}"
            );
        }
    }

    #[test]
    fn build_cost_groups_do_not_collapse_adams_tail() {
        let mut resolution = Resolution::new(299);
        let active_s = (0..=298).collect::<Vec<_>>();
        let basis_dims = vec![10; 300];
        let inner = ComputeMode::Accelerated {
            subalgebra: Subalgebra::a(0).unwrap(),
            strict: false,
            force: true,
        };
        let groups = resolution.fixed_t_batch_groups(
            299,
            &active_s,
            7,
            FixedTBatchScheduler::WeightedContiguousBuildCost,
            &basis_dims,
            &inner,
        );

        assert_eq!(groups.len(), 7);
        assert_eq!(
            groups.iter().flatten().copied().collect::<Vec<_>>(),
            active_s
        );
        let max_len = groups.iter().map(Vec::len).max().unwrap_or(0);
        assert!(
            max_len <= 50,
            "build-cost groups should not leave a collapsed high-s tail on one worker: {groups:?}"
        );
    }

    #[test]
    fn weighted_contiguous_groups_keep_zero_weight_tail_nonempty() {
        let values = vec![0, 1, 2, 3];
        let weights = vec![100, 0, 0, 0];
        let groups = weighted_contiguous_groups(&values, &weights, 3);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups, vec![vec![0], vec![1, 2], vec![3]]);
    }

    #[test]
    fn weighted_contiguous_groups_force_split_before_late_heavy_work() {
        let values = vec![0, 1, 2];
        let weights = vec![0, 0, 1000];
        let groups = weighted_contiguous_groups(&values, &weights, 3);
        assert_eq!(groups, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn round_robin_rank_groups_assign_by_s_mod_rank_count() {
        let values = vec![1, 3, 4, 7, 8, 12];
        let groups = round_robin_rank_groups(&values, 4);
        assert_eq!(
            groups,
            vec![vec![4, 8, 12], vec![1], Vec::<usize>::new(), vec![3, 7]]
        );
    }

    #[test]
    fn adams_vanishing_skip_uses_output_q_not_task_s() {
        assert!(!fixed_t_task_is_adams_vanishing(2, 8));
        assert!(fixed_t_task_is_adams_vanishing(3, 8));
        assert!(fixed_t_task_is_adams_vanishing(6, 8));
        assert!(!fixed_t_task_is_adams_vanishing(7, 8));

        assert!(!fixed_t_task_is_adams_vanishing(107, 324));
        assert!(fixed_t_task_is_adams_vanishing(108, 324));
        assert!(fixed_t_task_is_adams_vanishing(322, 324));
        assert!(!fixed_t_task_is_adams_vanishing(323, 324));
    }

    #[test]
    fn sparse_merge_preserves_caches_and_invalidates_changed_degree() {
        let base = Resolution::new(8);
        let base_snapshot = base.snapshot();
        let mut sparse =
            Resolution::from_sparse_snapshot(8, base.generator_count(), base_snapshot).unwrap();
        sparse
            .basis_cache
            .insert((0, 1), Arc::new(Vec::<BasisElem>::new()));
        sparse
            .basis_cache
            .insert((3, 1), Arc::new(Vec::<BasisElem>::new()));
        let mut history = HashMap::default();
        history.insert(0_usize, 100_u128);
        history.insert(1_usize, 200_u128);
        sparse.fixed_t_previous_task_micros = Some((7, history));
        sparse.fixed_t_previous_group_ranges = Some((7, vec![(0, 1)]));

        let next_id = sparse.generator_count();
        sparse
            .merge_sparse_snapshot(
                next_id + 1,
                ResolutionSnapshot {
                    generators: vec![Generator {
                        id: next_id,
                        s: 0,
                        t: 1,
                        differential: Arc::new(Vec::new()),
                    }],
                },
            )
            .unwrap();

        assert_eq!(sparse.generator_count(), next_id + 1);
        assert_eq!(sparse.generators[next_id].s, 0);
        assert!(!sparse.basis_cache.contains_key(&(0, 1)));
        assert!(sparse.basis_cache.contains_key(&(3, 1)));
        assert_eq!(sparse.fixed_t_previous_task_micros.as_ref().unwrap().0, 7);
        assert_eq!(
            sparse.fixed_t_previous_group_ranges.as_ref().unwrap().1,
            vec![(0, 1)]
        );
    }

    #[test]
    fn sparse_retain_prunes_unneeded_new_generators_without_resetting_history() {
        let mut resolution = Resolution::new(8);
        let xi1 = Milnor::new(vec![1]).packed().unwrap();
        let first_new = resolution.generator_count();
        let g1 = resolution.add_generator(
            1,
            1,
            vec![ModuleTerm {
                coeff_packed: xi1,
                generator: 0,
            }],
        );
        let g2 = resolution.add_generator(
            2,
            2,
            vec![ModuleTerm {
                coeff_packed: xi1,
                generator: g1,
            }],
        );
        let g3 = resolution.add_generator(
            3,
            3,
            vec![ModuleTerm {
                coeff_packed: xi1,
                generator: g2,
            }],
        );
        resolution
            .basis_cache
            .insert((1, 1), Arc::new(Vec::<BasisElem>::new()));
        resolution
            .basis_cache
            .insert((2, 2), Arc::new(Vec::<BasisElem>::new()));
        resolution
            .basis_cache
            .insert((4, 4), Arc::new(Vec::<BasisElem>::new()));
        let mut history = HashMap::default();
        history.insert(1_usize, 100_u128);
        resolution.fixed_t_previous_task_micros = Some((8, history));

        let meta_s = BTreeSet::from([1_usize, 2]);
        let diff_s = BTreeSet::from([1_usize]);
        let (removed, stripped) = resolution
            .retain_sparse_homological_degrees_from(first_new, &meta_s, &diff_s)
            .unwrap();

        assert_eq!((removed, stripped), (1, 1));
        assert_eq!(resolution.generators[g1].s, 1);
        assert!(!resolution.generators[g1].differential.is_empty());
        assert_eq!(resolution.generators[g2].s, 2);
        assert!(resolution.generators[g2].differential.is_empty());
        assert_eq!(resolution.generators[g3].s, usize::MAX);
        assert!(resolution.gens_by_s[1].contains(&g1));
        assert!(resolution.gens_by_s[2].contains(&g2));
        assert!(!resolution.gens_by_s[3].contains(&g3));
        assert!(resolution.basis_cache.contains_key(&(1, 1)));
        assert!(!resolution.basis_cache.contains_key(&(2, 2)));
        assert!(resolution.basis_cache.contains_key(&(4, 4)));
        assert_eq!(
            resolution.fixed_t_previous_task_micros.as_ref().unwrap().0,
            8
        );
    }

    #[test]
    fn build_cost_contiguous_scheduler_uses_matrix_build_weights() {
        let mut resolution = Resolution::new(8);
        let xi1 = Milnor::new(vec![1]).packed().unwrap();
        let xi2 = Milnor::new(vec![2]).packed().unwrap();
        let g1 = resolution.add_generator(
            1,
            1,
            vec![ModuleTerm {
                coeff_packed: xi1,
                generator: 0,
            }],
        );
        resolution.add_generator(
            2,
            2,
            vec![
                ModuleTerm {
                    coeff_packed: xi1,
                    generator: g1,
                },
                ModuleTerm {
                    coeff_packed: xi2,
                    generator: g1,
                },
                ModuleTerm {
                    coeff_packed: xi1,
                    generator: 0,
                },
            ],
        );
        let active_s = vec![0, 1, 2];
        let basis_dims = vec![1, 3, 5, 1];
        resolution.ensure_algebra_basis_through(5);
        let inner = ComputeMode::Accelerated {
            subalgebra: Subalgebra::a(0).unwrap(),
            strict: false,
            force: true,
        };
        let weights =
            resolution.fixed_t_scheduler_build_cost_task_weights(5, &active_s, &basis_dims, &inner);
        assert!(
            weights[1] > weights[0],
            "differential-length build cost should make s=1 heavier: {weights:?}"
        );

        let groups = resolution.fixed_t_batch_groups(
            5,
            &active_s,
            2,
            FixedTBatchScheduler::WeightedContiguousBuildCost,
            &basis_dims,
            &inner,
        );
        assert_eq!(groups, weighted_contiguous_groups(&active_s, &weights, 2));
    }

    #[test]
    fn fixed_t_local_s_chunks_leave_short_groups_unchanged() {
        let values = (13..=15).collect::<Vec<_>>();
        let basis_dims = vec![1; 20];
        let groups = fixed_t_local_s_groups(&values, &basis_dims, 8);
        let chunks = fixed_t_local_s_chunks(&values, &basis_dims, 8, 100);
        assert_eq!(groups, vec![vec![13], vec![14], vec![15]]);
        assert_eq!(chunks, groups);
    }

    #[test]
    fn fixed_t_local_s_chunks_use_singleton_queue_for_large_groups() {
        let values = (24..=281).collect::<Vec<_>>();
        let mut basis_dims = vec![1; 283];
        for dimension in &mut basis_dims[24..=35] {
            *dimension = 1000;
        }
        let chunks = fixed_t_local_s_chunks(&values, &basis_dims, 8, 282);
        assert!(
            chunks.iter().all(|chunk| chunk.len() == 1),
            "rank-local work should be singleton queue entries: {chunks:?}"
        );
        assert_eq!(chunks.len(), values.len(), "chunks={chunks:?}");
        let mut flattened = chunks.iter().flatten().copied().collect::<Vec<_>>();
        flattened.sort_unstable();
        assert_eq!(flattened, values);
        assert_eq!(
            chunks.first().and_then(|group| group.first()).copied(),
            Some(24)
        );
        assert_eq!(
            chunks.last().and_then(|group| group.last()).copied(),
            Some(281)
        );
    }

    #[test]
    fn fixed_t_local_s_chunks_are_independent_of_local_worker_count() {
        let values = (24..=281).collect::<Vec<_>>();
        let mut basis_dims = vec![1; 283];
        for dimension in &mut basis_dims[24..=35] {
            *dimension = 1000;
        }
        let local_s_workers = 16;
        let chunks = fixed_t_local_s_chunks(&values, &basis_dims, local_s_workers, 282);

        assert_eq!(chunks.len(), values.len(), "chunks={chunks:?}");
        assert!(chunks.iter().all(|chunk| chunk.len() == 1));

        let mut flattened = chunks.iter().flatten().copied().collect::<Vec<_>>();
        flattened.sort_unstable();
        assert_eq!(flattened, values);
    }

    #[test]
    fn fixed_t_local_s_chunks_rank_modulo_group_uses_local_queue() {
        let values = (35..=312).collect::<Vec<_>>();
        let mut basis_dims = vec![1; 314];
        for (offset, s) in (35..=52).enumerate() {
            basis_dims[s] = 1000usize.saturating_sub(offset * 40).max(100);
        }
        let chunks = fixed_t_local_s_chunks(&values, &basis_dims, 8, 313);

        assert_eq!(chunks.len(), values.len(), "chunks={chunks:?}");
        assert!(
            chunks.iter().all(|chunk| chunk.len() == 1),
            "rank-local work should use singleton local queue entries: {chunks:?}"
        );
        assert_eq!(chunks.iter().flatten().copied().collect::<Vec<_>>(), values);
    }

    #[test]
    fn take_new_generator_representatives_truncates_without_reordering() {
        let mut resolution = Resolution::new(8);
        let frozen_generator_count = resolution.generators.len();
        let first = vec![ModuleTerm {
            coeff_packed: 0,
            generator: 0,
        }];
        let second = vec![
            ModuleTerm {
                coeff_packed: 2,
                generator: 0,
            },
            ModuleTerm {
                coeff_packed: 1,
                generator: 0,
            },
        ];

        resolution.add_generator(1, 1, first.clone());
        resolution.add_generator(2, 2, second.clone());
        assert_eq!(resolution.generators.len(), frozen_generator_count + 2);

        let representatives = resolution
            .take_new_generator_representatives_and_truncate(
                frozen_generator_count,
                2,
                "test",
                0,
                0,
            )
            .unwrap();

        let mut expected_second = second;
        sort_terms(&mut expected_second);
        assert_eq!(representatives, vec![first, expected_second]);
        assert_eq!(resolution.generators.len(), frozen_generator_count);
        assert!(
            resolution
                .gens_by_s
                .get(1)
                .map(Vec::is_empty)
                .unwrap_or(true)
        );
        assert!(
            resolution
                .gens_by_s
                .get(2)
                .map(Vec::is_empty)
                .unwrap_or(true)
        );
    }

    #[test]
    fn fixed_t_local_s_chunk_count_uses_singletons() {
        assert_eq!(fixed_t_local_s_chunk_count(3, 8), 3);
        assert_eq!(fixed_t_local_s_chunk_count(8, 8), 8);
        assert_eq!(fixed_t_local_s_chunk_count(9, 8), 9);
        assert_eq!(fixed_t_local_s_chunk_count(64, 8), 64);
        assert_eq!(fixed_t_local_s_chunk_count(278, 8), 278);
        assert_eq!(fixed_t_local_s_chunk_count(278, 1), 278);
    }

    #[test]
    fn full_differential_chunk_override_restores_configured_limit() {
        let configured = configured_full_differential_chunk_bytes();
        assert_eq!(full_differential_chunk_bytes(), configured);

        {
            let _outer = scoped_full_differential_chunk_bytes_override(Some(7));
            assert_eq!(full_differential_chunk_bytes(), 7);

            {
                let _inner = scoped_full_differential_chunk_bytes_override(Some(11));
                assert_eq!(full_differential_chunk_bytes(), 11);
            }

            assert_eq!(full_differential_chunk_bytes(), 7);
        }

        assert_eq!(full_differential_chunk_bytes(), configured);
    }

    #[test]
    fn full_differential_output_chunk_override_batches_outputs() {
        assert_eq!(full_differential_output_chunk_bytes(), None);
        assert_eq!(full_differential_output_batch_len(512, 100), 100);

        {
            let _guard = scoped_full_differential_output_chunk_bytes_override(Some(1024));
            assert_eq!(full_differential_output_chunk_bytes(), Some(1024));
            assert_eq!(BitVec::storage_bytes_for_len(512), 64);
            assert_eq!(full_differential_output_batch_len(512, 100), 16);
            assert_eq!(full_differential_output_batch_len(4096, 100), 2);
            assert_eq!(full_differential_output_batch_len(1_000_000, 100), 1);
            assert_eq!(full_differential_output_batch_len(512, 0), 1);
        }

        assert_eq!(full_differential_output_chunk_bytes(), None);
    }

    #[test]
    fn profile_contiguous_groups_preserve_profile_runs_when_possible() {
        let values = (0..10).collect::<Vec<_>>();
        let weights = vec![3, 3, 3, 8, 8, 8, 8, 2, 2, 2];
        let profiles = vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2];
        let groups = profile_contiguous_groups(&values, &weights, &profiles, 5);
        assert_eq!(groups.len(), 5);
        for group in groups {
            let first_profile = profiles[group[0]];
            assert!(
                group.iter().all(|&s| profiles[s] == first_profile),
                "group {group:?} crosses a profile boundary"
            );
        }
    }

    #[test]
    fn profile_contiguous_groups_keep_exact_profile_runs() {
        let values = (0..8).collect::<Vec<_>>();
        let weights = vec![100, 100, 1, 1, 50, 50, 2, 2];
        let profiles = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let groups = profile_contiguous_groups(&values, &weights, &profiles, 4);
        assert_eq!(groups, vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]);
    }

    #[test]
    fn profile_contiguous_groups_merge_runs_when_workers_are_fewer() {
        let values = (0..6).collect::<Vec<_>>();
        let weights = vec![1; values.len()];
        let profiles = vec![0, 0, 1, 1, 2, 2];
        let groups = profile_contiguous_groups(&values, &weights, &profiles, 2);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups.concat(), values);
        for group in groups {
            for window in group.windows(2) {
                assert_eq!(window[0] + 1, window[1]);
            }
        }
    }

    #[test]
    fn adaptive_contiguous_groups_use_previous_task_times() {
        let active_s = vec![0, 1, 2, 3];
        let basis_dims = vec![1, 1, 1, 1, 0];
        let mut resolution = Resolution::new(4);
        let inner = ComputeMode::Naive;

        let fallback_groups = resolution.fixed_t_batch_groups(
            4,
            &active_s,
            2,
            FixedTBatchScheduler::AdaptiveContiguous,
            &basis_dims,
            &inner,
        );
        assert_eq!(fallback_groups, vec![vec![0, 1], vec![2, 3]]);

        let mut history = HashMap::<usize, u128>::default();
        history.insert(0, 200);
        history.insert(1, 1);
        history.insert(2, 1);
        history.insert(3, 1);
        resolution.fixed_t_previous_task_micros = Some((3, history));
        let adaptive_groups = resolution.fixed_t_batch_groups(
            4,
            &active_s,
            2,
            FixedTBatchScheduler::AdaptiveContiguous,
            &basis_dims,
            &inner,
        );
        assert_eq!(adaptive_groups, vec![vec![0], vec![1, 2, 3]]);
    }

    #[test]
    fn adaptive_minimax_contiguous_groups_use_previous_task_times() {
        let active_s = vec![0, 1, 2, 3, 4];
        let basis_dims = vec![1, 1, 1, 1, 1, 0];
        let mut resolution = Resolution::new(5);
        let inner = ComputeMode::Naive;
        let mut history = HashMap::<usize, u128>::default();
        history.insert(0, 1);
        history.insert(1, 1);
        history.insert(2, 1);
        history.insert(3, 1);
        history.insert(4, 5);
        resolution.fixed_t_previous_task_micros = Some((4, history));

        let groups = resolution.fixed_t_batch_groups(
            5,
            &active_s,
            2,
            FixedTBatchScheduler::AdaptiveMinimaxContiguous,
            &basis_dims,
            &inner,
        );
        assert_eq!(groups, vec![vec![0, 1, 2, 3], vec![4]]);
    }

    #[test]
    fn sticky_contiguous_groups_reuse_previous_ranges_when_balanced() {
        let values = (0..6).collect::<Vec<_>>();
        let weights = vec![1; values.len()];
        let previous_ranges = vec![(0, 1), (2, 3), (4, 4)];
        let groups = sticky_contiguous_groups(
            &values,
            &weights,
            3,
            Some(&previous_ranges),
            STICKY_CONTIGUOUS_MAX_OVER_BASE_NUM,
            STICKY_CONTIGUOUS_MAX_OVER_BASE_DEN,
            6,
        );
        assert_eq!(groups, vec![vec![0, 1], vec![2, 3], vec![4, 5]]);
    }

    #[test]
    fn sticky_contiguous_groups_fallback_when_previous_ranges_are_too_imbalanced() {
        let values = (0..8).collect::<Vec<_>>();
        let weights = vec![100, 100, 100, 100, 1, 1, 1, 1];
        let previous_ranges = vec![(0, 3), (4, 7)];
        let baseline = weighted_contiguous_groups(&values, &weights, 2);
        let groups = sticky_contiguous_groups(
            &values,
            &weights,
            2,
            Some(&previous_ranges),
            STICKY_CONTIGUOUS_MAX_OVER_BASE_NUM,
            STICKY_CONTIGUOUS_MAX_OVER_BASE_DEN,
            8,
        );
        assert_eq!(groups, baseline);
        assert_ne!(groups, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    }

    #[test]
    fn adaptive_sticky_contiguous_groups_use_previous_group_ranges() {
        let active_s = vec![0, 1, 2, 3, 4, 5];
        let basis_dims = vec![1, 1, 1, 1, 1, 1, 0];
        let mut resolution = Resolution::new(6);
        let inner = ComputeMode::Naive;
        let mut history = HashMap::<usize, u128>::default();
        for &s in &active_s {
            history.insert(s, 1);
        }
        resolution.fixed_t_previous_task_micros = Some((5, history));
        resolution.fixed_t_previous_group_ranges = Some((5, vec![(0, 1), (2, 3), (4, 4)]));

        let groups = resolution.fixed_t_batch_groups(
            6,
            &active_s,
            3,
            FixedTBatchScheduler::AdaptiveStickyContiguous,
            &basis_dims,
            &inner,
        );
        assert_eq!(groups, vec![vec![0, 1], vec![2, 3], vec![4, 5]]);
    }

    #[test]
    fn persistent_product_cache_assignment_prefers_group_overlap() {
        fn persisted(
            group_first: usize,
            group_last: usize,
            marker: CoeffKey,
        ) -> FixedTBatchPersistedProductCaches {
            let mut caches = FixedTBatchProductCaches::default();
            caches
                .mult_packed_cache
                .insert((marker, marker), vec![marker]);
            FixedTBatchPersistedProductCaches {
                group_first,
                group_last,
                caches,
            }
        }

        let mut stored = vec![
            Some(persisted(0, 4, 1)),
            Some(persisted(10, 14, 2)),
            Some(persisted(5, 9, 3)),
        ];

        let first = take_best_persistent_product_cache_for_group(&mut stored, &[6, 7, 8])
            .expect("overlapping cache should be selected");
        assert!(first.caches.mult_packed_cache.contains_key(&(3, 3)));

        let second = take_best_persistent_product_cache_for_group(&mut stored, &[12, 13])
            .expect("second overlapping cache should be selected");
        assert!(second.caches.mult_packed_cache.contains_key(&(2, 2)));
    }

    #[test]
    fn weighted_greedy_groups_split_heavy_prefix_tasks() {
        let values = (0..8).collect::<Vec<_>>();
        let weights = vec![100, 90, 1, 1, 1, 1, 1, 1];
        let groups = weighted_greedy_groups(&values, &weights, 2);
        let greedy_max = groups
            .iter()
            .map(|group| group.iter().map(|&s| weights[s]).sum::<u128>())
            .max()
            .unwrap();
        let contiguous_max = contiguous_groups(&values, 2)
            .iter()
            .map(|group| group.iter().map(|&s| weights[s]).sum::<u128>())
            .max()
            .unwrap();

        assert!(greedy_max < contiguous_max);
        assert_ne!(
            groups.iter().position(|group| group.contains(&0)),
            groups.iter().position(|group| group.contains(&1))
        );
    }

    #[test]
    fn weighted_front_merge_tail_split_merges_first_two_and_splits_last() {
        let values = (0..32).collect::<Vec<_>>();
        let weights = vec![1_u128; values.len()];
        let baseline = weighted_contiguous_groups(&values, &weights, 8);
        let groups = weighted_contiguous_front_merge_tail_split_groups(&values, &weights, 8);

        assert_eq!(groups.len(), 8);
        let mut expected_first = baseline[0].clone();
        expected_first.extend_from_slice(&baseline[1]);
        assert_eq!(groups[0], expected_first);
        for old_index in 2..baseline.len() - 1 {
            assert_eq!(groups[old_index - 1], baseline[old_index]);
        }

        let split_tail = groups[6..].iter().flatten().copied().collect::<Vec<_>>();
        assert_eq!(split_tail, *baseline.last().unwrap());
        assert_eq!(groups[6].len(), 3);
        assert_eq!(groups[7].len(), 1);
    }

    #[test]
    fn weighted_tail_split_two_to_one_uses_weights_not_count() {
        let values = vec![10, 11, 12, 13, 14];
        let weights = vec![9_u128, 1, 1, 1, 6];
        let groups = weighted_contiguous_tail_split_two_to_one(&values, &weights);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups.iter().flatten().copied().collect::<Vec<_>>(), values);
        let left_weight = group_weight(&groups[0], &values, &weights);
        let right_weight = group_weight(&groups[1], &values, &weights);
        assert!(
            left_weight >= right_weight,
            "rank 6 tail split should get the heavier 2:1 side: {groups:?}"
        );
    }

    #[test]
    fn frozen_view_excludes_current_t_and_is_immutable() {
        let mut resolution = Resolution::new(8);
        resolution.compute(3, ComputeMode::Naive).unwrap();

        let view = FrozenResolutionView::from_resolution(&resolution, 4);
        let before = view.build_basis(1);
        assert!(
            before
                .iter()
                .all(|elem| view.generator(elem.generator).t < 4)
        );

        let expected_len = resolution
            .gens_by_s
            .get(1)
            .into_iter()
            .flatten()
            .map(|&id| &resolution.generators[id])
            .filter(|generator| generator.t < 4)
            .map(|generator| resolution.algebra_basis[4 - generator.t].len())
            .sum::<usize>();
        assert_eq!(before.len(), expected_len);

        resolution.add_generator(1, 4, Vec::new());
        assert_eq!(view.build_basis(1), before);
        assert!(resolution.build_basis(1, 4).len() > before.len());
    }

    #[test]
    fn frozen_t_slice_differential_squares_to_zero() {
        let mut resolution = Resolution::new(8);
        resolution.compute(5, ComputeMode::Naive).unwrap();
        resolution.ensure_algebra_basis_through(6);
        let view = FrozenResolutionView::from_resolution(&resolution, 6);
        let mut builder = FrozenMatrixBuilder::new(&view);

        for s in 1..=5 {
            let d = builder.d_matrix(s).unwrap();
            let previous = builder.d_matrix(s - 1).unwrap();
            for column in &d.columns {
                let d2 = crate::f2::apply_columns(&previous.columns, previous.target_dim, column);
                assert!(d2.is_zero(), "frozen d^2 failed at s={s}");
            }
        }
    }

    #[test]
    fn fixed_t_batch_matches_naive_counts_and_validates_small_range() {
        let max_t = 8;
        let mut naive = Resolution::new(max_t);
        naive.compute(max_t, ComputeMode::Naive).unwrap();

        let mut batch = Resolution::new(max_t);
        batch
            .compute(
                max_t,
                ComputeMode::FixedTBatch {
                    workers: Some(1),
                    shadow: false,
                    validate_commit: true,
                    scheduler: FixedTBatchScheduler::Contiguous,
                    inner: Box::new(ComputeMode::Naive),
                },
            )
            .unwrap();

        assert_eq!(generator_counts(&batch), generator_counts(&naive));
        assert_resolution_valid_through(&mut batch, max_t);
        assert_exact_through(&mut batch, max_t);
    }

    #[test]
    fn fixed_t_batch_auto_workers_one_matches_naive_small_range() {
        let max_t = 8;
        let mut naive = Resolution::new(max_t);
        naive.compute(max_t, ComputeMode::Naive).unwrap();

        let mut batch = Resolution::new(max_t);
        batch
            .compute(
                max_t,
                ComputeMode::FixedTBatch {
                    workers: Some(1),
                    shadow: false,
                    validate_commit: true,
                    scheduler: FixedTBatchScheduler::Contiguous,
                    inner: Box::new(ComputeMode::Auto {
                        subalgebras: vec![
                            Subalgebra::a(2).unwrap(),
                            Subalgebra::a(1).unwrap(),
                            Subalgebra::a(0).unwrap(),
                        ],
                        strict: false,
                        force: false,
                    }),
                },
            )
            .unwrap();

        assert_eq!(generator_counts(&batch), generator_counts(&naive));
        assert_resolution_valid_through(&mut batch, max_t);
        assert_exact_through(&mut batch, max_t);
    }

    #[test]
    fn fixed_t_batch_auto_bounded_naive_refuses_large_uncovered_bidegree() {
        let mut resolution = Resolution::new(0);
        for _ in 0..30_000 {
            resolution.add_generator(0, 0, Vec::new());
            resolution.add_generator(1, 0, Vec::new());
        }
        let err = resolution
            .compute_step(
                1,
                0,
                &ComputeMode::AutoBoundedNaiveFallback {
                    subalgebras: Vec::new(),
                    force: false,
                },
            )
            .unwrap_err();
        assert!(
            err.contains("refuses large full naive fallback"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fixed_t_batch_is_deterministic_across_worker_counts() {
        let max_t = 7;
        let mut signatures = Vec::new();
        for workers in [1, 2, 4] {
            let mut resolution = Resolution::new(max_t);
            resolution
                .compute(
                    max_t,
                    ComputeMode::FixedTBatch {
                        workers: Some(workers),
                        shadow: false,
                        validate_commit: true,
                        scheduler: FixedTBatchScheduler::RoundRobin,
                        inner: Box::new(ComputeMode::Naive),
                    },
                )
                .unwrap();
            assert_resolution_valid_through(&mut resolution, max_t);
            signatures.push(resolution_signature(&resolution));
        }
        assert_eq!(signatures[1], signatures[0]);
        assert_eq!(signatures[2], signatures[0]);
    }

    #[test]
    fn fixed_t_batch_shadow_mode_checks_small_range() {
        let max_t = 7;
        let mut shadow = Resolution::new(max_t);
        shadow
            .compute(
                max_t,
                ComputeMode::FixedTBatch {
                    workers: Some(2),
                    shadow: true,
                    validate_commit: true,
                    scheduler: FixedTBatchScheduler::RoundRobin,
                    inner: Box::new(ComputeMode::Naive),
                },
            )
            .unwrap();

        let mut naive = Resolution::new(max_t);
        naive.compute(max_t, ComputeMode::Naive).unwrap();
        assert_eq!(generator_counts(&shadow), generator_counts(&naive));
    }

    fn generator_counts(resolution: &Resolution) -> BTreeMap<(usize, usize), usize> {
        let mut counts = BTreeMap::new();
        for generator in &resolution.generators {
            *counts.entry((generator.s, generator.t)).or_default() += 1;
        }
        counts
    }

    fn assert_resolution_valid_through(resolution: &mut Resolution, max_t: usize) {
        let ids = resolution
            .generators
            .iter()
            .filter(|generator| generator.t <= max_t)
            .map(|generator| generator.id)
            .collect::<Vec<_>>();
        for id in ids {
            resolution.verify_minimal_generator(id).unwrap();
            resolution.verify_generator_d_squared(id).unwrap();
        }
    }

    fn assert_exact_through(resolution: &mut Resolution, max_t: usize) {
        for t in 0..=max_t {
            for s in 0..=t {
                let homology = resolution.naive_homology_representatives(s, t).unwrap();
                assert!(homology.is_empty(), "homology remains at (s,t)=({s},{t})");
            }
        }
    }

    type ResolutionSignature = Vec<(usize, usize, Vec<(CoeffKey, usize)>)>;

    fn resolution_signature(resolution: &Resolution) -> ResolutionSignature {
        resolution
            .generators
            .iter()
            .map(|generator| {
                (
                    generator.s,
                    generator.t,
                    generator
                        .differential
                        .iter()
                        .map(|term| (term.coeff_packed, term.generator))
                        .collect(),
                )
            })
            .collect()
    }
}
