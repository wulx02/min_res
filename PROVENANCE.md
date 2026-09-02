# Code provenance and upstream relationships

This project was implemented by OpenAI Codex at the direction of the repository
owner. During development, Codex translated or adapted some upstream code and
used other upstream code, designs, and mathematical methods as references. The
function-level relationships are listed below.

The labels used here have distinct meanings:

- **Direct source adaptation:** recognizable source code or control flow was
  translated and modified.
- **Structural adaptation:** an upstream function's stages or data flow were
  retained, but the implementation was substantially rewritten.
- **Implementation reference:** an upstream representation, interface, or
  implementation pattern guided a different local implementation.
- **Shared mathematical algorithm:** the functions implement the same published
  mathematics; this label alone does not claim a source-code translation.

## `src/fast_hash.rs`

### `FastHasher::mix`

- Upstream function: Sebastiano Vigna's
  [SplitMix64 `next`](https://prng.di.unimi.it/splitmix64.c).
- Relationship: **direct source adaptation** for the initial SplitMix64 output
  mixing sequence: the additive constant, two shift-and-multiply steps, and
  final XOR-shift. The subsequent combination into `FastHasher::state` is a
  local extension.
- Upstream status: Vigna's 2015 source dedicates the code to the public domain
  to the extent possible under law and otherwise grants permission to use,
  copy, modify, and distribute it for any purpose.

## `src/milnor.rs`

### `Milnor::degree`, `weight`, and `Milnor` display

- Upstream functions used as references: SSeqCpp `MMilnor::deg` and
  `MMilnor::Str`, and SpectralSequences/sseq's `MilnorBasisElement` display and
  `xi_degrees`-based degree handling.
- Relationship: **implementation reference** and **shared mathematical
  algorithm**. These are the standard degree and exponent-vector presentation
  of the Milnor basis; no distinctive upstream function body is retained.

### `max_mask`

- Upstream function: SSeqCpp
  [`max_mask`](https://github.com/WayneLin92/SSeqCpp/blob/23d12c973db2b294a6c00c15bd106e70b0af3fa6/src/steenrod.cpp).
- Relationship: **direct source adaptation**. The shift-and-mask body is a Rust
  translation of the SSeqCpp function.

### `define_mul_packed_xi_v3_for_each!` and its generated functions

This entry covers `mul_packed_xi_v3_for_each_1` through
`mul_packed_xi_v3_for_each_9`.

- Upstream function: SSeqCpp
  [`MulMilnorV3`](https://github.com/WayneLin92/SSeqCpp/blob/23d12c973db2b294a6c00c15bd106e70b0af3fa6/src/steenrod.cpp).
- Relationship: **direct source adaptation**. The `X`, `XR`, `XS`, and `XT`
  state, `R_floor`, initialization, mask search, traversal, backtracking, and
  result construction follow `MulMilnorV3`.
- Local changes include Rust translation, widths 1 through 9, a different
  packed coefficient layout, callback-based term emission, bounded-degree
  dispatch, and Rust arithmetic and indexing conventions.

### Fast-multiplication callers

This entry covers `multiply_packed_fast`,
`multiply_packed_fast_bounded_matching`, `multiply_packed_fast_raw`,
`multiply_packed_fast_raw_for_each`,
`multiply_packed_fast_raw_for_each_bounded_degree`, and
`multiply_packed_fast_raw_for_each_width`.

- Upstream functions: SSeqCpp `MulMilnor`, `MulMilnorV3`, `Milnor::operator*`,
  and `mulP` in
  [`src/steenrod.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/23d12c973db2b294a6c00c15bd106e70b0af3fa6/src/steenrod.cpp).
- Relationship: these are local dispatch and filtering wrappers around the
  directly adapted V3 kernel. Their wrapper bodies are not translations of a
  single upstream function, but their multiplication work is performed by the
  adapted kernel above.

### Packed exponent conversion

This entry covers `pack_entries`, `pack_padded_entries`,
`pack_padded_entries_unchecked`, `packed_entry`, `packed_entry_mask`,
`pack_xi_1` through `pack_xi_9`, `unpack_xi_1` through `unpack_xi_9`,
`unpack_packed_entries_trimmed`, `milnor_from_packed`, `Milnor::packed`, and
`Milnor::from_packed`.

- Upstream functions used as references:
  - SSeqCpp `MMilnor::Xi` and `MMilnor::ToXi` in
    [`include/algebras/steenrod.h`](https://github.com/WayneLin92/SSeqCpp/blob/23d12c973db2b294a6c00c15bd106e70b0af3fa6/include/algebras/steenrod.h).
  - SpectralSequences/sseq `PPart::try_from_slice`, `PPart::from_slice`,
    `PPart::set`, `PPart::get`, `PPart::bits`, and `PPart::from_bits` in
    [`milnor_algebra.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/crates/algebra/src/algebra/milnor_algebra.rs).
- Relationship: **implementation reference**. All three implementations store
  a Milnor exponent vector in one machine word and convert between packed and
  unpacked forms. This project uses its own contiguous field widths and is not
  byte-compatible with either upstream representation.

### Milnor-basis generation

This entry covers `basis_through_degree`, `basis_keys_through_degree`,
`basis_of_degree`, `basis_keys_of_degree`,
`basis_keys_of_degree_with_capacity`, `packed_basis_counts_through_degree`,
`packed_basis_count_of_degree`, `basis_rec`, `basis_keys_rec`, and
`max_milnor_index`.

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorAlgebra::compute_ppart` and `MilnorAlgebra::generate_basis_2` in
  [`milnor_algebra.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/crates/algebra/src/algebra/milnor_algebra.rs).
- Relationship: **implementation reference** and **shared mathematical
  algorithm**. The local code recursively enumerates weighted exponent
  partitions; sseq incrementally extends degree-indexed tables. The local
  function bodies are not translations of those two functions.

### Generic Milnor multiplication

This entry covers `multiply_packed_entries_with_row_cache_internal`,
`multiply_rec`, `row_decompositions`, `row_decomp_rec`, and
`leaf_term_packed`.

- Upstream functions used as references: SpectralSequences/sseq
  `PPartMultiplier::new_from_allocation`, `PPartMultiplier::next_val`,
  `PPartMultiplier::update`, and `PPartMultiplier::next` in
  [`milnor_algebra.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/crates/algebra/src/algebra/milnor_algebra.rs).
- Relationship: **structural adaptation** and **shared mathematical
  algorithm**. Both represent Milnor multiplication by constrained matrices,
  enumerate admissible entries, reject diagonals whose binary summands
  overlap, form output exponents from diagonal sums, and cancel repeated terms
  modulo two.
- Local changes: this project precomputes and caches weighted row
  decompositions, traverses them recursively instead of mutating sseq's matrix
  iterator, retains only column sums and diagonal state during traversal, uses
  the project's packed coefficient layout, and adds callback filters and
  optional profile-trivial filtering. A local hash set performs parity
  cancellation as terms are emitted.

### `sort_packed_mod2`

- Upstream function: SSeqCpp
  [`SortMod2(MMilnor1d&)`](https://github.com/WayneLin92/SSeqCpp/blob/23d12c973db2b294a6c00c15bd106e70b0af3fa6/src/steenrod.cpp).
- Relationship: **structural adaptation**. Both first sort sparse terms and
  then remove equal terms in pairs to reduce coefficients modulo two.
- Local changes: this project scans each complete equal run, writes back only
  odd-parity runs, truncates the vector in place, and uses its packed Milnor
  ordering rather than SSeqCpp's `MMilnor` ordering and container operations.

## `src/f2.rs`

### `XorBasis::reduce` and `XorBasis::insert`

- Upstream functions used as references:
  - SSeqCpp `Residue`, `ResidueInplace`, `AddToSpace`, and `GetSpace` in
    [`src/linalg.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/23d12c973db2b294a6c00c15bd106e70b0af3fa6/src/linalg.cpp).
  - SpectralSequences/sseq `Subspace::reduce`, `Subspace::add_vector`, and
    `Matrix::row_reduce` in its
    [`matrix`](https://github.com/SpectralSequences/sseq/tree/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/crates/fp/src/matrix)
    module.
- Relationship: **structural adaptation** of pivot reduction and basis
  insertion. The local representation is a dense `u64` bit vector; SSeqCpp
  uses sorted sparse indices and sseq uses its general finite-field matrix
  types.

### `ImageBasis::reduce`, `ImageBasis::insert`, and `ImageBasis::insert_or_relation`

- Upstream functions used as references:
  - SSeqCpp `GetInvMap`, `SetLinearMap`, `SetLinearMapV2`, and
    `SetLinearMapV3` in
    [`src/linalg.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/23d12c973db2b294a6c00c15bd106e70b0af3fa6/src/linalg.cpp).
  - SpectralSequences/sseq `Matrix::row_reduce`,
    `Matrix::compute_quasi_inverse`, and `Matrix::compute_kernel` in
    [`matrix_inner.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/crates/fp/src/matrix/matrix_inner.rs).
- Relationship: **structural adaptation**. The local functions reduce an image
  while applying the same row operations to a source combination; a zero image
  produces a relation. The storage and APIs are local.

### `kernel_with_label`

- Upstream functions used as references: SSeqCpp `SetLinearMap` and
  SpectralSequences/sseq `Matrix::compute_kernel`.
- Relationship: **structural adaptation**. Unit source vectors are tracked
  while image columns are inserted, and dependencies become kernel vectors.
  The logging and dense-bit-vector implementation are local.

### `LinearSolver::new`, `LinearSolver::solve`, and `ImageBasis::solve`

- Upstream functions used as references:
  - SSeqCpp `GetInvMap`, `GetImage`, and `GetInvImage` in
    [`src/linalg.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/23d12c973db2b294a6c00c15bd106e70b0af3fa6/src/linalg.cpp).
  - SpectralSequences/sseq `Matrix::compute_quasi_inverse` and
    `QuasiInverse::apply` in its
    [`matrix`](https://github.com/SpectralSequences/sseq/tree/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/crates/fp/src/matrix)
    module.
- Relationship: **structural adaptation** of an image basis carrying chosen
  preimages. It is not a translation of sseq's matrix or quasi-inverse types.

### `quotient_representatives`

- Upstream functions used as references: SSeqCpp `QuotientSpace` and
  SpectralSequences/sseq `Subquotient::from_parts`.
- Relationship: **structural adaptation** of reducing one subspace modulo
  another and retaining independent representatives.

### `apply_columns`

- Upstream functions used as references: SSeqCpp `GetImage` and
  SpectralSequences/sseq `Matrix::apply`.
- Relationship: **implementation reference**. The operation is ordinary matrix
  application over `F_2`; the local body XORs selected dense columns.

### Homology representative functions

This entry covers `homology_representatives`,
`homology_representatives_with_label`,
and `homology_representative_batches_with_label`.

- Upstream functions used as references: SpectralSequences/sseq
  `Subquotient::from_parts`, `Subspace::reduce`, `Matrix::compute_kernel`, and
  the kernel/image construction in its resolution functions.
- Relationship: **structural adaptation** of computing
  `kernel(d) / image(d_next)`. Batching and memory instrumentation are local.

## `src/subalgebra.rs`

### Subalgebra constructors

This entry covers `Subalgebra::a`, `Subalgebra::b_profile`, `Subalgebra::f`,
`Subalgebra::fprime`, `Subalgebra::from_profile`, and
`Subalgebra::from_signatures`.

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorSubalgebra::new` and `MilnorSubalgebra::zero_algebra` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/nassau.rs).
- Relationship: **implementation reference** for representing a finite
  subalgebra by a profile. The local constructors additionally support the
  named `A`, `B`, `F`, and `F'` families.

### `Subalgebra::profile_tau`

- Upstream function: SpectralSequences/sseq `MilnorSubalgebra::top_degree` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/nassau.rs).
- Relationship: **structural adaptation**. Both sum
  `(2^profile[i] - 1)(2^(i+1) - 1)` over the profile; the local function adds
  checked shifts and supports its broader profile representation.

### Packed-signature functions

This entry covers `split_profile_signature_packed`,
`profile_signature_is_zero_packed_unchecked`, and `signature_packed`.

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorSubalgebra::packed_signature` and
  `MilnorSubalgebra::signature_mask` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/nassau.rs).
- Relationship: **structural adaptation**. As in sseq, these functions classify
  a Milnor basis element by selecting the profile-controlled low bits of each
  packed exponent coordinate and use the resulting signature in filtration
  tests.
- Local changes: the masks are applied to this project's fixed-field packed
  layout rather than sseq's compiled `(mask, value)` representation;
  `split_profile_signature_packed` also returns the complementary quotient;
  wide profile entries are handled explicitly; and `signature_packed` extends
  the representation to the local `F` and `F'` families.

### Signature enumeration and ordering

This entry covers `generate_signatures`, `compatible_bit_order`,
`signature_key`, `sort_signatures`, and `compare_signature_order`.

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorSubalgebra::iter_signatures`, `SignatureIterator::new`, and
  `SignatureIterator::next`, together with the signature order documented in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/nassau.rs).
- Relationship: **structural adaptation** and **shared mathematical
  algorithm**. The local `A` and `B` signatures retain the same mixed-radix
  precedence: the first Milnor coordinate varies fastest, coordinate values
  increase from zero, and the profile and total-degree bounds limit them.
- Local changes: this project recursively materializes all signatures up to an
  explicit degree bound and then sorts them, whereas sseq advances a mutable
  mixed-radix iterator. The local ordering is represented by an explicit bit
  list and is extended to the `F` and `F'` families.

### Lower-line functions

This entry covers `profile_d`, `profile_lower_ok`, `lower_line_applies`, and
`lower_line_bound`.

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorSubalgebra::top_degree` and `MilnorSubalgebra::optimal_for` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/nassau.rs).
- Relationship: **implementation reference** and **shared mathematical
  algorithm** for the published Nassau lower-line bounds.
- Local changes: this project evaluates the formulas for its general profile
  representation, adds checked arithmetic, and extends the conditions to its
  `A`, `B`, `F`, and `F'` families and detailed-support data.

## `src/resolution.rs`

### `Resolution::choose_subalgebra`

- Upstream function: SpectralSequences/sseq
  `MilnorSubalgebra::optimal_for` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/nassau.rs).
- Relationship: **structural adaptation** of the candidate-selection stage.
  The upstream function traverses an ordered iterator and selects the last
  candidate in its initial consecutive applicable prefix with
  `take_while(...).last()`.
- Local changes: this project accepts an explicit candidate list, applies
  disable, force, and certification rules, estimates cost from the three
  adjacent zero-signature basis dimensions, and resolves ties with a local
  family priority instead of retaining the terminal candidate from that
  upstream prefix.

### Free-module basis functions

This entry covers `Resolution::build_basis` and
`FrozenResolutionView::build_basis`.

- Upstream functions used as references: SpectralSequences/sseq
  `FreeModule::compute_basis`, `FreeModule::iter_gens`,
  `FreeModule::iter_gen_offsets`, `FreeModule::generator_offset`,
  `FreeModule::operation_generator_to_index`, and
  `FreeModule::index_to_op_gen` in
  [`free_module.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/crates/algebra/src/module/free_module.rs).
- Relationship: **structural adaptation**. Like sseq, the functions traverse
  graded free generators in generator order, compute the complementary algebra
  degree, and append the algebra basis in its established order.
- Local changes: this project materializes `(packed coefficient, global
  generator id)` as `BasisElem` values instead of maintaining sseq's offset
  tables and index-conversion API. The frozen view also excludes generators
  from the layer currently being computed and shares immutable data through
  `Arc`.

### Differential application and matrix construction

This entry covers both implementations of
`differential_of_basis_elem_packed`, both `d_matrix` functions, and
`d_matrix_signature`.

- Upstream functions used as references: SpectralSequences/sseq
  `FreeModuleHomomorphism::apply_to_basis_element`,
  `ModuleHomomorphism::get_matrix`,
  `ModuleHomomorphism::get_partial_matrix`, and
  `MilnorSubalgebra::signature_matrix` in
  [`free_module_homomorphism.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/crates/algebra/src/module/homomorphism/free_module_homomorphism.rs),
  [`homomorphism/mod.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/crates/algebra/src/module/homomorphism/mod.rs),
  and [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/nassau.rs).
- Relationship: **structural adaptation**. Both form a matrix by applying the
  stored image of each free generator to every free-module basis element, then
  select signature-specific domain and target bases and construct the
  restricted matrix.
- Local changes: this project multiplies packed coefficients, looks up target
  rows through `(coefficient, generator)` maps, stores dense `F_2` columns, and
  adds product, basis, routing, and matrix caches. Signature columns can be
  built in parallel, and the frozen implementation reads an immutable layer
  snapshot. The local code also reports missing terms and signature-order
  violations explicitly.

### `Resolution::step_algorithm2`

- Upstream function: SpectralSequences/sseq
  `Resolution::step_resolution_with_subalgebra` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/nassau.rs).
- Relationship: **structural adaptation** and **shared mathematical algorithm**.
  The local function retains the upstream function's main stages: compute
  signature-zero homology, turn its representatives into candidate generator
  images, apply the full differential, iterate over nonzero signatures, solve
  each lifting problem, update images and errors, verify that the errors vanish,
  and add the new generators.
- The local body substantially rewrites those stages around dense column
  vectors, batching, signature translation, cache reuse, memory limits,
  profiling, and error reporting. It is not a line-by-line translation, but the
  function-level control flow is recognizably the same.

### Naive resolution functions

This entry covers `step_naive` and `naive_homology_representatives`.

- Upstream function used as a reference: SpectralSequences/sseq
  `Resolution::step_resolution` in
  [`ext/src/resolution.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/resolution.rs).
- Relationship: **implementation reference** and **shared mathematical
  algorithm**. Both add free generators that kill the relevant homology. The
  local sphere-resolution path computes `kernel(d) / image(d_next)` directly
  and is not a translation of sseq's augmented chain-map construction.

### Signature-restricted helpers used by `step_algorithm2`

This entry covers `basis_signature_cached`, `coeff_signature_basis_cached`,
`basis_signature_index_cached`, `basis_signature_routing_cached`, and
`extract_signature_vector`.

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorSubalgebra::signature_mask`, `MilnorSubalgebra::signature_matrix`, and
  `ModuleHomomorphism::get_partial_matrix`.
- Relationship: **structural adaptation**. These local helpers split and cache
  pieces that are computed inline by sseq's
  `step_resolution_with_subalgebra`.

### Lifting helpers used by `step_algorithm2`

This entry covers `linear_solver_cached`, `solve_signature_lifts`, and
`signature_to_zero_translation`.

- Upstream functions used as references: SpectralSequences/sseq
  `Matrix::compute_quasi_inverse`, `QuasiInverse::apply`, and their use inside
  `step_resolution_with_subalgebra`.
- Relationship: **structural adaptation**. The local solver stores a pivoted
  image with source combinations and can translate a signature problem to the
  zero-signature problem before solving it.

### Correction helpers used by `step_algorithm2`

This entry covers `apply_full_differentials_to_vectors` and
`signature_vector_to_terms`.

- Upstream code used as a reference: the `xs` and `dxs` construction and
  correction loops inside SpectralSequences/sseq
  `Resolution::step_resolution_with_subalgebra`.
- Relationship: **structural adaptation**. sseq performs these operations
  inline; this project separates and batches them.

### `Resolution::add_generator`

- Upstream functions used as references: SpectralSequences/sseq
  `Resolution::add_generators`, `FreeModule::add_generators`, and
  `FreeModuleHomomorphism::add_generators_from_rows`.
- Relationship: **implementation reference**. The local function combines
  generator metadata and its differential in one project-specific record.

### Fixed-internal-degree parallel layer functions

This entry covers `compute_from_cursor_fixed_t_batch_with_progress`,
`compute_fixed_t_batch_shadow_layer`, `compute_fixed_t_batch_layer`, and the
`compute_isolated_bidegree_group*` family.

- Upstream functions used as references:
  - SSeqCpp `Resolve` in
    [`Adams/groebner_res.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/23d12c973db2b294a6c00c15bd106e70b0af3fa6/Adams/groebner_res.cpp).
  - SpectralSequences/sseq `Resolution::compute_through_bidegree_with_callback`
    and its Nassau resolution computation loop in
    [`ext/src/resolution.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/resolution.rs)
    and [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/nassau.rs).
- Relationship: **structural adaptation** for SSeqCpp's outer fixed-`t` loop,
  parallel per-`s` work, layer barrier, and post-barrier commit; and
  **implementation reference** for sseq's dependency-aware parallel resolution
  scheduling. The Rayon worker groups, persistent caches, load-balancing
  heuristics, memory controls, and Grid orchestration are local extensions.

### Sequential computation loop

This entry covers `compute_from_cursor_with_progress`.

- Upstream functions used as references:
  - The Nassau implementation of SpectralSequences/sseq
    `Resolution::compute_through_bidegree` in
    [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/nassau.rs).
  - The ordinary implementation of SpectralSequences/sseq
    `Resolution::compute_through_bidegree_with_callback` in
    [`ext/src/resolution.rs`](https://github.com/SpectralSequences/sseq/blob/e1e0f6f30ce56855c71793808577d9369a5c2f21/ext/src/resolution.rs).
- Relationship: **structural adaptation** of the Nassau function's
  internal-degree-major traversal with homological degree in the inner loop.
  The ordinary callback function is an **implementation reference** only for
  notifying the caller after a bidegree; its dependency-aware scheduler is not
  retained here.
- Local changes: this project resumes from a checkpoint cursor, grows the
  Milnor basis and subalgebra signatures one internal-degree layer at a time,
  enforces its triangular task bound, and dispatches its own computation modes
  and fallback policies. The callback receives project-specific progress and
  checkpoint state.

## Published Nassau algorithm and `cnassau/steenrod`

- Paper: Christian Nassau,
  [“Computing a minimal resolution over the Steenrod algebra”](https://arxiv.org/abs/1910.04063).
- Related implementation:
  [cnassau/steenrod](https://github.com/cnassau/steenrod).

`Subalgebra::profile_tau`, `Subalgebra::profile_d`, the lower-line tests,
signature decomposition, and `Resolution::step_algorithm2` implement Nassau's
published signature-filtration method. The GPL-2.0-licensed
`cnassau/steenrod` project is cited as a related implementation. No source code
from it is copied, translated, linked, or distributed here; the local code
provenance is the paper and the SSeqCpp and SpectralSequences/sseq functions
identified above.

## Licenses

SSeqCpp is by Weinan Lin and is licensed under the Apache License 2.0.
SpectralSequences/sseq is licensed under MIT or Apache License 2.0; this
distribution uses its Apache License 2.0 option for derived material. See
`LICENSE-APACHE` and `THIRD_PARTY_NOTICES.md`.
