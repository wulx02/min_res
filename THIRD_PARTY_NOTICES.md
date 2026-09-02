# Third-party notices

## SSeqCpp

This project contains a Rust adaptation of part of SSeqCpp:

- Project: SSeqCpp
- Author and copyright holder: Weinan Lin
- Upstream repository: <https://github.com/WayneLin92/SSeqCpp>
- Audited upstream revision: `23d12c973db2b294a6c00c15bd106e70b0af3fa6`
- License: Apache License 2.0; see [LICENSE-APACHE](LICENSE-APACHE)

The adapted code is the optimized Milnor-basis multiplication
matrix-enumeration kernel corresponding to SSeqCpp's `max_mask` and
`MulMilnorV3` in `src/steenrod.cpp`. In this repository it appears in
`src/milnor.rs`, beginning at `max_mask` and including the
`define_mul_packed_xi_v3_for_each!` implementation and its width-specific
instantiations.

The code has been translated from C++ to Rust and modified to use this
project's packed Milnor coefficient representation, widths 1 through 9,
callback-based term emission, bounded-degree dispatch, and Rust arithmetic and
indexing conventions. The surrounding basis construction, generic reference
multiplication, Nassau minimal-resolution implementation, finite-subalgebra
logic, linear algebra, checkpoint format, exporters, and parallel/Grid runtime
are not translations of SSeqCpp in the source comparison performed for this
notice.

SSeqCpp's upstream license contains the following attribution:

> Copyright 2023 Weinan Lin
