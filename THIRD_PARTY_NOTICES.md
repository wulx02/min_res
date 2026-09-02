# Third-party notices

## SSeqCpp

This project's Steenrod-algebra implementation was informed by and partly
adapted from SSeqCpp:

- Project: SSeqCpp
- Author and copyright holder: Weinan Lin
- Upstream repository: <https://github.com/WayneLin92/SSeqCpp>
- License: Apache License 2.0; see [LICENSE-APACHE](LICENSE-APACHE)

The relationship includes the following parts:

- The compact representation of Milnor-basis monomials and conversion between
  exponent vectors and packed machine words were informed by SSeqCpp's
  `MMilnor`, `Xi`, and `ToXi` design. This repository uses a different bit
  layout: nine contiguous exponent fields rather than SSeqCpp's exponent-bit
  and May-weight layout.
- The optimized Milnor-basis multiplication matrix-enumeration kernel is a
  direct Rust adaptation of SSeqCpp's `max_mask` and `MulMilnorV3` in
  `src/steenrod.cpp`. In this repository it appears in `src/milnor.rs`,
  beginning at `max_mask` and including the
  `define_mul_packed_xi_v3_for_each!` implementation and its width-specific
  instantiations.
- The fixed-internal-degree organization of the computation, with independent
  homological-degree tasks processed in parallel before a layer barrier, was
  informed by the parallel organization of SSeqCpp's Adams-resolution
  computation.

SSeqCpp's upstream license contains the following attribution:

> Copyright 2023 Weinan Lin
