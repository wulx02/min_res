# Code provenance and upstream relationships

This project was implemented by OpenAI Codex at the direction of the repository
owner. During development, Codex directly translated some upstream source and
structurally adapted some upstream functions and designs. The function-level
relationships are listed below.

The labels used here have distinct meanings:

- **Direct source adaptation:** recognizable source-level logic or detailed
  control flow was translated and modified.
- **Structural adaptation:** an upstream function's higher-level organization,
  stages, or data flow were retained, but the implementation was rewritten.

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

### `max_mask`

- Upstream function: SSeqCpp
  [`max_mask`](https://github.com/WayneLin92/SSeqCpp/blob/7942e10aee1193bfbc42227752c55c7429f43523/src/steenrod.cpp).
- Relationship: **direct source adaptation**. The shift-and-mask body is a Rust
  translation of the SSeqCpp function.

### `define_mul_packed_xi_v3_for_each!` and its generated functions

This entry covers `mul_packed_xi_v3_for_each_1` through
`mul_packed_xi_v3_for_each_9`.

- Upstream function: SSeqCpp
  [`MulMilnorV3`](https://github.com/WayneLin92/SSeqCpp/blob/7942e10aee1193bfbc42227752c55c7429f43523/src/steenrod.cpp).
- Relationship: **direct source adaptation**. The `X`, `XR`, `XS`, and `XT`
  state, `R_floor`, initialization, mask search, traversal, backtracking, and
  result construction follow `MulMilnorV3`.
- Local changes include Rust translation, widths 1 through 9, a different
  packed coefficient layout, callback-based term emission, bounded-degree
  dispatch, and Rust arithmetic and indexing conventions.
- Local callers `multiply_packed_fast`, `multiply_packed_fast_bounded_matching`,
  `multiply_packed_fast_raw`, `multiply_packed_fast_raw_for_each`,
  `multiply_packed_fast_raw_for_each_bounded_degree`, and
  `multiply_packed_fast_raw_for_each_width` provide dispatch and filtering
  around the adapted kernel. Their width selection, filtering, and callback
  bodies are local; the conversion flow retained from SSeqCpp is recorded in
  the next entry.
- Runtime role: `multiply_packed_fast_bounded_matching`,
  `multiply_packed_fast_raw_for_each_bounded_degree`, and
  `multiply_packed_fast_raw_for_each_width` form the production bulk-product
  path. The unbounded `multiply_packed_fast`, `multiply_packed_fast_raw`, and
  `multiply_packed_fast_raw_for_each` wrappers are compiled only for tests.

### Packed Milnor representation and packing/unpacking functions

This entry covers `PACKED_ENTRY_WIDTHS`, `M0` through `M8`, `Milnor::packed`,
`Milnor::from_packed`, `pack_entries`, `pack_padded_entries`,
`pack_padded_entries_unchecked`, `packed_entry`, `packed_entry_mask`,
`unpack_xi_1` through `unpack_xi_9`, `pack_xi_1` through `pack_xi_9`,
`unpack_packed_entries_trimmed`, `milnor_from_packed`, and the conversion flow
in `multiply_packed_fast_raw_for_each` and
`multiply_packed_fast_raw_for_each_width`.

- Upstream functions and design:
  - SSeqCpp
    [`MMilnor::Xi` and `MMilnor::ToXi`](https://github.com/WayneLin92/SSeqCpp/blob/7942e10aee1193bfbc42227752c55c7429f43523/include/algebras/steenrod.h),
    together with their use around
    [`MulMilnorV3` in `MulMilnor`](https://github.com/WayneLin92/SSeqCpp/blob/7942e10aee1193bfbc42227752c55c7429f43523/src/steenrod.cpp).
  - SpectralSequences/sseq
    [`MilnorHashMap::code`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/crates/algebra/src/algebra/milnor_algebra.rs),
    which packs position-dependent Milnor exponent fields into a `u64` lookup
    key.
- Relationship: **structural adaptation**. As in SSeqCpp, this project stores
  Milnor basis elements in a `u64`, converts packed operands to exponent arrays
  before the V3 multiplication kernel, and converts each resulting exponent
  array back to the packed representation. This is the local
  compression/decompression boundary. As in sseq's lookup key, the local
  representation packs exponent coordinates into position-dependent fields.
- Local changes: SSeqCpp decomposes each exponent into binary
  `xi_j^(2^i)` generator-incidence bits, stores those bits in a 37-bit field,
  and stores May weight in a separate 9-bit field. This project instead stores
  nine exponent coordinates directly in contiguous fields of widths
  `[10, 8, 7, 6, 5, 4, 3, 2, 1]`; it neither uses SSeqCpp's generator lookup
  tables nor stores May weight. Unlike sseq's degree-specific lookup key, this
  project retains the first exponent so the word can be decoded without being
  given its degree, and it uses field widths chosen for its degree-512 bound.
  The three formats are not binary-compatible. The local functions also add
  checked packing, inverse decoding, trailing-zero trimming, widths 1 through
  9, bounded-degree dispatch, and callback-based result emission.

### Generic Milnor multiplication

This entry covers `multiply_packed_entries_with_row_cache_internal`,
`multiply_rec`, `row_decompositions`, `row_decomp_rec`, and
`leaf_term_packed`.

- Upstream functions: SpectralSequences/sseq
  `PPartMultiplier::new_from_allocation`, `PPartMultiplier::next_val`,
  `PPartMultiplier::update`, and `PPartMultiplier::next` in
  [`milnor_algebra.rs`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/crates/algebra/src/algebra/milnor_algebra.rs).
- Relationship: **structural adaptation**. Both represent Milnor multiplication
  by constrained matrices, enumerate admissible entries, reject diagonals whose
  binary summands overlap, form output exponents from diagonal sums, and cancel
  repeated terms modulo two.
- Runtime role: this generic implementation is not the main multiplication
  path in default large computations. The default fixed-`t` Algorithm 2 path
  performs its bulk multiplication with the adapted V3 kernel above. The
  generic implementation is retained mainly for the `ext multiply` command,
  low-dimensional naive or fallback computation, optional commit-time
  `d^2 = 0` validation, and independent cross-checking in tests.
- Local changes: this project precomputes and caches weighted row
  decompositions, traverses them recursively instead of mutating sseq's matrix
  iterator, retains only column sums and diagonal state during traversal, uses
  the project's packed coefficient layout, and adds callback filters and
  optional profile-trivial filtering. A local hash set performs parity
  cancellation as terms are emitted.

### `sort_packed_mod2`

- Upstream function: SSeqCpp
  [`SortMod2(MMilnor1d&)`](https://github.com/WayneLin92/SSeqCpp/blob/7942e10aee1193bfbc42227752c55c7429f43523/src/steenrod.cpp).
- Relationship: **structural adaptation**. Both first sort sparse terms and
  then remove equal terms in pairs to reduce coefficients modulo two.
- Runtime role: the production bounded V3 wrapper
  `multiply_packed_fast_bounded_matching` uses this function to normalize
  duplicate terms emitted by the V3 kernel. The generic multiplication above
  performs its parity cancellation separately with a hash set.
- Local changes: this project scans each complete equal run, writes back only
  odd-parity runs, truncates the vector in place, and uses its packed Milnor
  ordering rather than SSeqCpp's `MMilnor` ordering and container operations.

## `src/f2.rs`

This file contains a single local dense-bit-vector linear-algebra layer, with
separate structures for pivot reduction, image/preimage tracking, and solving.
Where both SSeqCpp and sseq are named below, they are two upstream sources for
the structure of this layer, not alternative runtime backends.

### `XorBasis::reduce`

- Upstream functions:
  - SSeqCpp `Residue` and `ResidueInplace` in
    [`src/linalg.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/7942e10aee1193bfbc42227752c55c7429f43523/src/linalg.cpp).
  - SpectralSequences/sseq `Subspace::reduce` in its
    [`matrix`](https://github.com/SpectralSequences/sseq/tree/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/crates/fp/src/matrix)
    module.
- Relationship: **structural adaptation** of reduction by an ordered pivot
  basis. The local representation is a dense `u64` bit vector; SSeqCpp uses
  sorted sparse indices and sseq uses its general finite-field matrix types.

### `XorBasis::insert`

- Upstream functions: SSeqCpp `AddToSpace` and `GetSpace` in
  [`src/linalg.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/7942e10aee1193bfbc42227752c55c7429f43523/src/linalg.cpp).
- Relationship: **structural adaptation** of SSeqCpp's incremental insertion.
  The local function reduces one dense bit vector and installs only its new
  pivot row.

### `ImageBasis::reduce`, `ImageBasis::insert`, and `ImageBasis::insert_or_relation`

- Upstream functions:
  - SSeqCpp `GetInvMap`, `SetLinearMap`, `SetLinearMapV2`, and
    `SetLinearMapV3` in
    [`src/linalg.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/7942e10aee1193bfbc42227752c55c7429f43523/src/linalg.cpp).
  - SpectralSequences/sseq `Matrix::row_reduce`,
    `Matrix::compute_quasi_inverse`, and `Matrix::compute_kernel` in
    [`matrix_inner.rs`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/crates/fp/src/matrix/matrix_inner.rs).
- Relationship: **structural adaptation**. The local functions reduce an image
  while applying the same row operations to a source combination; a zero image
  produces a relation. The storage and APIs are local.

### `kernel_with_label`

- Upstream functions: SSeqCpp `SetLinearMap` and
  SpectralSequences/sseq `Matrix::compute_kernel`.
- Relationship: **structural adaptation**. Unit source vectors are tracked
  while image columns are inserted, and dependencies become kernel vectors.
  The logging and dense-bit-vector implementation are local.

### `LinearSolver::new`, `LinearSolver::solve`, and `ImageBasis::solve`

- Upstream functions:
  - SSeqCpp `GetInvMap`, `GetImage`, and `GetInvImage` in
    [`src/linalg.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/7942e10aee1193bfbc42227752c55c7429f43523/src/linalg.cpp).
  - SpectralSequences/sseq `Matrix::compute_quasi_inverse` and
    `QuasiInverse::apply` in its
    [`matrix`](https://github.com/SpectralSequences/sseq/tree/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/crates/fp/src/matrix)
    module.
- Relationship: **structural adaptation** of an image basis carrying chosen
  preimages. The local code uses dense `BitVec` storage and project-specific
  solver types and APIs.

### `quotient_representatives`

- Upstream functions: SSeqCpp `QuotientSpace` and
  SpectralSequences/sseq `Subquotient::from_parts`.
- Relationship: **structural adaptation** of reducing one subspace modulo
  another and retaining independent representatives.
- Runtime role: this helper is compiled only for tests. Production homology
  computation uses the functions in the following section.

### Homology representative functions

This entry covers `homology_representatives`,
`homology_representatives_with_label`,
and `homology_representative_batches_with_label`.

- Upstream functions: SpectralSequences/sseq
  `Subquotient::from_parts`, `Subspace::reduce`, `Matrix::compute_kernel`, and
  the kernel/image construction in its resolution functions.
- Relationship: **structural adaptation** of computing
  `kernel(d) / image(d_next)`. Batching and memory instrumentation are local.
- Runtime role: `homology_representative_batches_with_label` contains the
  shared implementation. The other two functions are convenience wrappers
  that collect all batches and optionally omit diagnostic labels; they are not
  separate homology algorithms.

## `src/subalgebra.rs`

### `Subalgebra::profile_tau`

- Upstream function: SpectralSequences/sseq `MilnorSubalgebra::top_degree` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/src/nassau.rs).
- Relationship: **structural adaptation**. Both sum
  `(2^profile[i] - 1)(2^(i+1) - 1)` over the profile; the local function adds
  checked shifts and supports its broader profile representation.

### Packed-signature functions

This entry covers `split_profile_signature_packed`,
`profile_signature_is_zero_packed_unchecked`, and `signature_packed`.

- Upstream functions: SpectralSequences/sseq
  `MilnorSubalgebra::has_signature` and `MilnorSubalgebra::signature_mask` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/src/nassau.rs).
- Relationship: **structural adaptation**. Both classify a Milnor basis element
  by the profile-controlled low bits of each exponent coordinate and use that
  classification to select signature-specific basis elements.
- Local changes: the code in the cited sseq revision compares an unpacked
  exponent vector with a requested signature. This project instead extracts a
  signature in its fixed-field packed layout;
  `split_profile_signature_packed` also returns the complementary quotient,
  the zero-signature test avoids materializing a signature, wide profile
  entries are handled explicitly, and `signature_packed` extends the operation
  to the local `F` and `F'` families.

### Signature enumeration and ordering

This entry covers `generate_signatures`, `compatible_bit_order`,
`profile_signature_index_packed`, `signature_key`, `sort_signatures`, and
`compare_signature_order`.

- Upstream functions: SpectralSequences/sseq
  `MilnorSubalgebra::iter_signatures`, `SignatureIterator::new`, and
  `SignatureIterator::next`, together with the signature order documented in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/src/nassau.rs).
- Relationship: **structural adaptation**. The local `A` and `B` signatures
  retain the same mixed-radix precedence: the first Milnor coordinate varies
  fastest, coordinate values increase from zero, and the profile and
  total-degree bounds limit them.
- Local changes: this project recursively materializes all signatures up to an
  explicit degree bound and then sorts them, whereas sseq advances a mutable
  mixed-radix iterator. The local ordering is represented by an explicit bit
  list and `profile_signature_index_packed` converts that list to a numeric
  index; the ordering is also extended to the `F` and `F'` families.

## `src/resolution.rs`

### Free-module basis functions

This entry covers `Resolution::build_basis` and
`FrozenResolutionView::build_basis`.

- Upstream functions: SpectralSequences/sseq
  `FreeModule::compute_basis`, `FreeModule::iter_gens`,
  `FreeModule::iter_gen_offsets`, `FreeModule::generator_offset`,
  `FreeModule::operation_generator_to_index`, and
  `FreeModule::index_to_op_gen` in
  [`free_module.rs`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/crates/algebra/src/module/free_module.rs).
- Relationship: **structural adaptation**. Like sseq, the functions traverse
  graded free generators in generator order, compute the complementary algebra
  degree, and append the algebra basis in its established order.
- Local changes: this project materializes `(packed coefficient, global
  generator id)` as `BasisElem` values instead of maintaining sseq's offset
  tables and index-conversion API. The frozen view also excludes generators
  from the layer currently being computed and shares immutable data through
  `Arc`.
- Runtime role: `Resolution::build_basis` is the normal implementation. The
  frozen-view implementation is used only by explicitly selected fixed-`t`
  naive computation and tests; it is not used by the default
  large-computation path.

### Differential application and matrix construction

This entry covers both implementations of
`differential_of_basis_elem_packed`, both `d_matrix` functions, and
`d_matrix_signature`.

- Upstream functions: SpectralSequences/sseq
  `FreeModuleHomomorphism::apply_to_basis_element`,
  `ModuleHomomorphism::get_matrix`,
  `ModuleHomomorphism::get_partial_matrix`, and
  `MilnorSubalgebra::signature_matrix` in
  [`free_module_homomorphism.rs`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/crates/algebra/src/module/homomorphism/free_module_homomorphism.rs),
  [`homomorphism/mod.rs`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/crates/algebra/src/module/homomorphism/mod.rs),
  and [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/src/nassau.rs).
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
- Runtime role: the `Resolution` implementations serve normal computation.
  The `FrozenMatrixBuilder` implementations serve the same opt-in fixed-`t`
  naive path described above and tests; they are not a second default matrix
  backend.

### `Resolution::step_algorithm2`

- Upstream function: SpectralSequences/sseq
  `Resolution::step_resolution_with_subalgebra` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/src/nassau.rs).
- Relationship: **structural adaptation**. The local function retains the
  upstream function's main stages: compute
  signature-zero homology, turn its representatives into candidate generator
  images, apply the full differential, iterate over nonzero signatures, solve
  each lifting problem, update images and errors, verify that the errors vanish,
  and add the new generators.
- The local body rewrites those stages around dense column vectors, batching,
  signature translation, cache reuse, memory limits, profiling, and error
  reporting. The function-level control flow remains recognizably the same.
- Runtime role: this is the usual per-bidegree mathematical algorithm selected
  inside the default fixed-`t` auto mode whenever an eligible subalgebra is
  available. Fixed-`t` is the outer scheduler, not an alternative to Algorithm
  2; low-dimensional bidegrees may instead use the naive fallback.

### Signature-restricted helpers used by `step_algorithm2`

This entry covers `basis_signature_cached`, `coeff_signature_basis_cached`,
`basis_signature_index_cached`, `basis_signature_routing_cached`, and
`extract_signature_vector`.

- Upstream functions: SpectralSequences/sseq
  `MilnorSubalgebra::signature_mask`, `MilnorSubalgebra::signature_matrix`, and
  `ModuleHomomorphism::get_partial_matrix`.
- Relationship: **structural adaptation**. These local helpers split and cache
  pieces that are computed inline by sseq's
  `step_resolution_with_subalgebra`.

### Lifting helpers used by `step_algorithm2`

This entry covers `linear_solver_cached`, `solve_signature_lifts`, and
`signature_to_zero_translation`.

- Upstream functions: SpectralSequences/sseq
  `Matrix::compute_quasi_inverse`, `QuasiInverse::apply`, and their use inside
  `step_resolution_with_subalgebra`.
- Relationship: **structural adaptation**. The local solver stores a pivoted
  image with source combinations and can translate a signature problem to the
  zero-signature problem before solving it.

### Correction helpers used by `step_algorithm2`

This entry covers `apply_full_differentials_to_vectors` and
`signature_vector_to_terms`.

- Upstream code: the `xs` and `dxs` construction and
  correction loops inside SpectralSequences/sseq
  `Resolution::step_resolution_with_subalgebra`.
- Relationship: **structural adaptation**. sseq performs these operations
  inline; this project separates and batches them.

### Fixed-internal-degree parallel layer functions

This entry covers `compute_from_cursor_fixed_t_batch_with_progress`,
`compute_fixed_t_batch_shadow_layer`, `compute_fixed_t_batch_layer`, and the
`compute_isolated_bidegree_group*` family.

- Upstream function: SSeqCpp `Resolve` in
  [`Adams/groebner_res.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/7942e10aee1193bfbc42227752c55c7429f43523/Adams/groebner_res.cpp).
- Relationship: **structural adaptation** of SSeqCpp's outer fixed-`t` loop,
  parallel per-`s` work, layer barrier, and post-barrier commit. The Rayon
  worker groups, persistent caches, load-balancing heuristics, memory controls,
  and Grid orchestration are local extensions.
- Runtime role: the ordinary fixed-`t` branch (`shadow = false`) implements the
  default computation mode. `compute_fixed_t_batch_shadow_layer` is an
  explicitly requested diagnostic that compares a batch layer with a
  sequential one.

### Sequential computation loop

This entry covers `compute_from_cursor_with_progress`.

- Upstream function: the Nassau implementation of SpectralSequences/sseq
  `Resolution::compute_through_bidegree` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext/src/nassau.rs).
- Relationship: **structural adaptation** of the Nassau function's
  internal-degree-major traversal with homological degree in the inner loop.
- Local changes: this project resumes from a checkpoint cursor, grows the
  Milnor basis and subalgebra signatures one internal-degree layer at a time,
  enforces its triangular task bound, and dispatches its own computation modes
  and fallback policies. The callback receives project-specific progress and
  checkpoint state.
- Runtime role: this is the common compute entry point. In the default
  fixed-`t` mode it delegates immediately to the fixed-`t` implementation
  above; its own sequential loop runs for explicitly selected non-batch modes
  and as a reference inside shadow diagnostics.

## Published Nassau algorithm

- Paper: Christian Nassau,
  [“Computing a minimal resolution over the Steenrod algebra”](https://arxiv.org/abs/1910.04063).

`Subalgebra::profile_tau`, `Subalgebra::profile_d`, the lower-line tests,
signature decomposition, and `Resolution::step_algorithm2` implement Nassau's
published signature-filtration method.

## Licenses

SSeqCpp is by Weinan Lin and is licensed under the Apache License 2.0.
SpectralSequences/sseq is licensed under MIT or Apache License 2.0; this
distribution uses its Apache License 2.0 option for derived material. See
`LICENSE-APACHE` and `THIRD_PARTY_NOTICES.md`.
