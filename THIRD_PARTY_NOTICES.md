# Third-party notices

This project was implemented by OpenAI Codex. During development, Codex directly
translated or structurally adapted parts of upstream code and designs. A
detailed function-by-function account is in [PROVENANCE.md](PROVENANCE.md).

## 1. SSeqCpp

Codex directly adapted part of the Milnor multiplication code and structurally
adapted additional SSeqCpp functions. The local-to-upstream function mapping is
in [PROVENANCE.md](PROVENANCE.md).

- Project: [SSeqCpp](https://github.com/WayneLin92/SSeqCpp)
- Author and copyright holder: Weinan Lin
- License: Apache License 2.0; see [LICENSE-APACHE](LICENSE-APACHE)

The directly adapted code is in `src/milnor.rs`: `max_mask` and the
`define_mul_packed_xi_v3_for_each!` kernel follow SSeqCpp's `max_mask` and
`MulMilnorV3`. They were translated to Rust and modified for this project's
representation and interfaces.

Codex also structurally adapted SSeqCpp's packed Milnor conversion pattern: the
`MMilnor::Xi`/`MMilnor::ToXi` packing/unpacking
(compression/decompression) boundary around the multiplication kernel. Other
structural adaptations cover sparse `F_2` normalization, linear algebra, and
fixed-internal-degree computation. The exact relationships and the locally
different packed representation are recorded in [PROVENANCE.md](PROVENANCE.md)
and in comments near the affected source.

SSeqCpp's upstream license contains the following notice:

> Copyright 2023 Weinan Lin

The SSeqCpp-derived portions have been modified and are distributed under the
Apache License 2.0.

## 2. SpectralSequences/sseq

Codex structurally adapted code and designs from the `ext` implementation in
the SpectralSequences/sseq repository, including its position-dependent packed
Milnor lookup-key design.

- Project: [SpectralSequences/sseq](https://github.com/SpectralSequences/sseq),
  especially its [`ext`](https://github.com/SpectralSequences/sseq/tree/ac6f59d751307439a9ccc05ef6f08d9eea22e3dd/ext)
  directory
- Authors named in the `ext` package metadata: Hood Chatham, Dexter Chua, and
  Joey Beauvais-Feisthauer
- License: MIT or Apache License 2.0

The local-to-upstream function mapping, including the relationship assigned to
each entry, is in [PROVENANCE.md](PROVENANCE.md). Other upstream contributors
retain attribution for their contributions. This distribution uses the Apache
License 2.0 option for material derived from SpectralSequences/sseq.

## 3. Nassau algorithm

The resolution algorithm is based on Christian Nassau's paper,
[“Computing a minimal resolution over the Steenrod algebra”](https://arxiv.org/abs/1910.04063).
See [PROVENANCE.md](PROVENANCE.md) for the code-provenance mapping and the
functions that implement the published algorithm.

## 4. SplitMix64 output mixer

The initial output-mixing sequence in `src/fast_hash.rs` function
`FastHasher::mix` is adapted from Sebastiano Vigna's
[SplitMix64 implementation](https://prng.di.unimi.it/splitmix64.c). The
subsequent combination into this project's hash state is local.

The upstream source was written by Sebastiano Vigna in 2015 and dedicated to
the public domain to the extent possible under law; it also grants permission
to use, copy, modify, and distribute the software for any purpose.
