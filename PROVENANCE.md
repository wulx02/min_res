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
  [`max_mask`](https://github.com/WayneLin92/SSeqCpp/blob/master/src/steenrod.cpp).
- Relationship: **direct source adaptation**. The shift-and-mask body is a Rust
  translation of the SSeqCpp function.

### `define_mul_packed_xi_v3_for_each!` and its generated functions

This entry covers `mul_packed_xi_v3_for_each_1` through
`mul_packed_xi_v3_for_each_9`.

- Upstream function: SSeqCpp
  [`MulMilnorV3`](https://github.com/WayneLin92/SSeqCpp/blob/master/src/steenrod.cpp).
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
  [`src/steenrod.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/master/src/steenrod.cpp).
- Relationship: these are local dispatch and filtering wrappers around the
  directly adapted V3 kernel. Their wrapper bodies are not translations of a
  single upstream function, but their multiplication work is performed by the
  adapted kernel above.

### Packed exponent conversion

This entry covers `pack_entries`, `pack_padded_entries`,
`pack_padded_entries_unchecked`, `packed_entry`, `packed_entry_mask`,
`pack_xi_1` through `pack_xi_9`, `unpack_xi_1` through `unpack_xi_9`,
`Milnor::packed`, and `Milnor::from_packed`.

- Upstream functions used as references:
  - SSeqCpp `MMilnor::Xi` and `MMilnor::ToXi` in
    [`include/algebras/steenrod.h`](https://github.com/WayneLin92/SSeqCpp/blob/master/include/algebras/steenrod.h).
  - SpectralSequences/sseq `PPart::try_from_slice`, `PPart::from_slice`,
    `PPart::set`, `PPart::get`, `PPart::bits`, and `PPart::from_bits` in
    [`milnor_algebra.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/crates/algebra/src/algebra/milnor_algebra.rs).
- Relationship: **implementation reference**. All three implementations store
  a Milnor exponent vector in one machine word and convert between packed and
  unpacked forms. This project uses its own contiguous field widths and is not
  byte-compatible with either upstream representation.

### Milnor-basis generation

This entry covers `basis_of_degree`, `basis_keys_of_degree`,
`basis_keys_of_degree_with_capacity`, `packed_basis_counts_through_degree`,
`basis_rec`, and `basis_keys_rec`.

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorAlgebra::compute_ppart` and `MilnorAlgebra::generate_basis_2` in
  [`milnor_algebra.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/crates/algebra/src/algebra/milnor_algebra.rs).
- Relationship: **implementation reference** and **shared mathematical
  algorithm**. The local code recursively enumerates weighted exponent
  partitions; sseq incrementally extends degree-indexed tables. The local
  function bodies are not translations of those two functions.

### Generic Milnor multiplication

This entry covers `multiply_packed_entries_with_row_cache_internal`,
`multiply_rec`, `row_decompositions`, `row_decomp_rec`, and
`leaf_term_packed`, together with the public `multiply*_with_row_cache`
wrappers that call them.

- Upstream functions used as references: SpectralSequences/sseq
  `PPartMultiplier::new_from_allocation`, `PPartMultiplier::next_val`,
  `PPartMultiplier::update`, and `PPartMultiplier::next` in
  [`milnor_algebra.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/crates/algebra/src/algebra/milnor_algebra.rs).
- Relationship: **implementation reference** and **shared mathematical
  algorithm**. Both enumerate matrices in Milnor's product formula and cancel
  coefficients modulo two. The local code uses cached recursive row
  decompositions rather than sseq's mutable matrix iterator, so this is not a
  line-by-line translation.

### `sort_packed_mod2`

- Upstream function: SSeqCpp
  [`SortMod2(MMilnor1d&)`](https://github.com/WayneLin92/SSeqCpp/blob/master/src/steenrod.cpp).
- Relationship: **implementation reference**. Both sort sparse terms and cancel
  equal terms modulo two. The local implementation counts the parity of each
  complete run instead of marking and removing adjacent pairs.

## `src/f2.rs`

### `XorBasis::reduce` and `XorBasis::insert`

- Upstream functions used as references:
  - SSeqCpp `Residue`, `ResidueInplace`, `AddToSpace`, and `GetSpace` in
    [`src/linalg.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/master/src/linalg.cpp).
  - SpectralSequences/sseq `Subspace::reduce`, `Subspace::add_vector`, and
    `Matrix::row_reduce` in its
    [`matrix`](https://github.com/SpectralSequences/sseq/tree/master/ext/crates/fp/src/matrix)
    module.
- Relationship: **structural adaptation** of pivot reduction and basis
  insertion. The local representation is a dense `u64` bit vector; SSeqCpp
  uses sorted sparse indices and sseq uses its general finite-field matrix
  types.

### `ImageBasis::reduce`, `ImageBasis::insert`, and `ImageBasis::insert_or_relation`

- Upstream functions used as references:
  - SSeqCpp `GetInvMap`, `SetLinearMap`, `SetLinearMapV2`, and
    `SetLinearMapV3` in
    [`src/linalg.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/master/src/linalg.cpp).
  - SpectralSequences/sseq `Matrix::row_reduce`,
    `Matrix::compute_quasi_inverse`, and `Matrix::compute_kernel` in
    [`matrix_inner.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/crates/fp/src/matrix/matrix_inner.rs).
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
    [`src/linalg.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/master/src/linalg.cpp).
  - SpectralSequences/sseq `Matrix::compute_quasi_inverse` and
    `QuasiInverse::apply` in its
    [`matrix`](https://github.com/SpectralSequences/sseq/tree/master/ext/crates/fp/src/matrix)
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
`homology_representative_batches_with_label`, and
`HomologyRepresentativeBatches::next_batch`.

- Upstream functions used as references: SpectralSequences/sseq
  `Subquotient::from_parts`, `Subspace::reduce`, `Matrix::compute_kernel`, and
  the kernel/image construction in its resolution functions.
- Relationship: **structural adaptation** of computing
  `kernel(d) / image(d_next)`. Batching and memory instrumentation are local.

## `src/subalgebra.rs`

### `Subalgebra::a`, `Subalgebra::b_profile`, and `Subalgebra::from_profile`

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorSubalgebra::new` and `MilnorSubalgebra::zero_algebra` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/src/nassau.rs).
- Relationship: **implementation reference** for representing a finite
  subalgebra by a profile. The local constructors additionally support the
  named `A`, `B`, `F`, and `F'` families.

### `Subalgebra::profile_tau`

- Upstream function: SpectralSequences/sseq `MilnorSubalgebra::top_degree` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/src/nassau.rs).
- Relationship: **structural adaptation**. Both sum
  `(2^profile[i] - 1)(2^(i+1) - 1)` over the profile; the local function adds
  checked shifts and supports its broader profile representation.

### Packed-signature functions

This entry covers `split_profile_signature_packed`,
`profile_quotient_packed_unchecked`,
`profile_signature_is_zero_packed_unchecked`,
`attach_profile_signature_packed_unchecked`, `signature_packed`,
`signature_index_packed`, and `profile_signature_index_packed`, together with
their test-only checked wrappers.

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorSubalgebra::packed_signature` and
  `MilnorSubalgebra::signature_mask` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/src/nassau.rs).
- Relationship: **implementation reference**. Both exploit the low profile
  bits of packed Milnor exponents to classify a signature. The local functions
  also split, reattach, and directly index signatures, which are not operations
  provided by the two upstream functions.

### Signature enumeration and ordering

This entry covers `generate_signatures`, `compatible_bit_order`,
`signature_key`, `sort_signatures`, and `compare_signature_order`.

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorSubalgebra::iter_signatures`, `SignatureIterator::new`, and
  `SignatureIterator::next`, together with the signature order documented in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/src/nassau.rs).
- Relationship: **implementation reference** and **shared mathematical
  algorithm**. The local enumeration is recursive and then explicitly sorted;
  sseq uses a mixed-radix iterator.

### Lower-line and subalgebra selection functions

This entry covers `profile_d`, `profile_lower_ok`, `lower_line_applies`,
`lower_line_bound`, `Resolution::choose_subalgebra`, and
`Resolution::selected_algorithm2_subalgebra_for_mode`.

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorSubalgebra::top_degree` and `MilnorSubalgebra::optimal_for` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/src/nassau.rs).
- Relationship: **implementation reference** and **shared mathematical
  algorithm**. The local selection supports more subalgebra families,
  certification tables, priorities, and cost estimates than the upstream
  function.

## `src/resolution.rs`

### Free-module basis functions

This entry covers `Resolution::build_basis`, `Resolution::basis_cached`, and
`FrozenResolutionView::build_basis`.

- Upstream functions used as references: SpectralSequences/sseq
  `FreeModule::compute_basis`, `FreeModule::iter_gens`,
  `FreeModule::iter_gen_offsets`, `FreeModule::generator_offset`,
  `FreeModule::operation_generator_to_index`, and
  `FreeModule::index_to_op_gen` in
  [`free_module.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/crates/algebra/src/module/free_module.rs).
- Relationship: **implementation reference** for ordering algebra-basis
  elements over graded free generators. The local project stores this mapping
  in `BasisElem` values and caches it differently.

### Differential application and matrix construction

This entry covers both implementations of
`differential_of_basis_elem_packed`, both `d_matrix` functions, and
`d_matrix_signature`.

- Upstream functions used as references: SpectralSequences/sseq
  `FreeModuleHomomorphism::apply_to_basis_element`,
  `ModuleHomomorphism::get_matrix`,
  `ModuleHomomorphism::get_partial_matrix`, and
  `MilnorSubalgebra::signature_matrix` in
  [`free_module_homomorphism.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/crates/algebra/src/module/homomorphism/free_module_homomorphism.rs),
  [`homomorphism/mod.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/crates/algebra/src/module/homomorphism/mod.rs),
  and [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/src/nassau.rs).
- Relationship: **implementation reference**. Both obtain a differential matrix
  by applying the stored image of each free generator and then restrict it to a
  signature. The local implementation uses packed coefficients, column
  matrices, and project-specific caches.

### `Resolution::step_algorithm2`

- Upstream function: SpectralSequences/sseq
  `Resolution::step_resolution_with_subalgebra` in
  [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/src/nassau.rs).
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
  [`ext/src/resolution.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/src/resolution.rs).
- Relationship: **implementation reference** and **shared mathematical
  algorithm**. Both add free generators that kill the relevant homology. The
  local sphere-resolution path computes `kernel(d) / image(d_next)` directly
  and is not a translation of sseq's augmented chain-map construction.

### Signature-restricted helpers used by `step_algorithm2`

This entry covers `d_matrix_signature_cached`, `basis_signature_cached`,
`coeff_signature_basis_cached`, `basis_signature_index_cached`,
`basis_signature_routing_cached`, and `extract_signature_vector`.

- Upstream functions used as references: SpectralSequences/sseq
  `MilnorSubalgebra::signature_mask`, `MilnorSubalgebra::signature_matrix`, and
  `ModuleHomomorphism::get_partial_matrix`.
- Relationship: **structural adaptation**. These local helpers split and cache
  pieces that are computed inline by sseq's
  `step_resolution_with_subalgebra`.

### Lifting helpers used by `step_algorithm2`

This entry covers `linear_solver_cached` and `solve_signature_lifts`.

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
`compute_fixed_t_batch_layer`, and the
`compute_isolated_bidegree_group*` family.

- Upstream functions used as references:
  - SSeqCpp `Resolve` in
    [`Adams/groebner_res.cpp`](https://github.com/WayneLin92/SSeqCpp/blob/master/Adams/groebner_res.cpp).
  - SpectralSequences/sseq `Resolution::compute_through_bidegree_with_callback`
    and its Nassau resolution computation loop in
    [`ext/src/resolution.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/src/resolution.rs)
    and [`ext/src/nassau.rs`](https://github.com/SpectralSequences/sseq/blob/master/ext/src/nassau.rs).
- Relationship: **structural adaptation** for SSeqCpp's outer fixed-`t` loop,
  parallel per-`s` work, layer barrier, and post-barrier commit; and
  **implementation reference** for sseq's dependency-aware parallel resolution
  scheduling. The Rayon worker groups, persistent caches, load-balancing
  heuristics, memory controls, and Grid orchestration are local extensions.

### Sequential computation loop

This entry covers `compute_from_cursor_with_progress` and `compute_step`.

- Upstream functions used as references: SpectralSequences/sseq
  `Resolution::compute_through_bidegree` and
  `Resolution::compute_through_bidegree_with_callback` in its ordinary and
  Nassau resolution implementations.
- Relationship: **implementation reference**. The local functions traverse the
  triangular `(s,t)` range, resume from a cursor, and dispatch between the naive
  and signature-filtered steps. The checkpoint cursor and per-layer Milnor
  basis growth are local behavior.

## Published Nassau algorithm and `cnassau/steenrod`

- Paper: Christian Nassau,
  [“Computing a minimal resolution over the Steenrod algebra”](https://arxiv.org/abs/1910.04063).
- Related implementation:
  [cnassau/steenrod](https://github.com/cnassau/steenrod).

`Subalgebra::profile_tau`, `Subalgebra::profile_d`, the lower-line tests,
signature decomposition, and `Resolution::step_algorithm2` implement Nassau's
published signature-filtration method. Codex also used `cnassau/steenrod` as an
implementation reference for profile selection, multiplication matrices,
signature handling, and the resolution loop. This is recorded as mathematical
and implementation-reference use, not as a direct source adaptation.

## Licenses

SSeqCpp is by Weinan Lin and is licensed under the Apache License 2.0.
SpectralSequences/sseq is licensed under MIT or Apache License 2.0; this
distribution uses its Apache License 2.0 option for derived material. See
`LICENSE-APACHE` and `THIRD_PARTY_NOTICES.md`.
