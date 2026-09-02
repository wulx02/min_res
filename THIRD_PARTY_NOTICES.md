# Third-party notices

This project was implemented by OpenAI Codex. During development, Codex
translated or adapted some upstream code and used other code
and designs as references. A detailed function-by-function account is in
[PROVENANCE.md](PROVENANCE.md).

## 1. SSeqCpp

Codex directly adapted part of the Milnor multiplication code and used
additional SSeqCpp functions as structural or implementation references. The
local-to-upstream function mapping is in [PROVENANCE.md](PROVENANCE.md).

- Project: [SSeqCpp](https://github.com/WayneLin92/SSeqCpp)
- Author and copyright holder: Weinan Lin
- License: Apache License 2.0; see [LICENSE-APACHE](LICENSE-APACHE)

The clearest direct adaptation is the optimized Milnor multiplication code in
`src/milnor.rs`: `max_mask` and the
`define_mul_packed_xi_v3_for_each!` kernel follow SSeqCpp's `max_mask` and
`MulMilnorV3`. They were translated to Rust and modified for this project's
representation and interfaces.

Codex also used SSeqCpp functions for packed Milnor conversion, sparse `F_2`
normalization, linear algebra, and fixed-internal-degree computation as
references. The exact relationship is recorded function by function in
[PROVENANCE.md](PROVENANCE.md) and in comments near the affected source.

SSeqCpp's upstream license contains the following notice:

> Copyright 2023 Weinan Lin

The SSeqCpp-derived portions have been modified and are distributed under the
Apache License 2.0.

## 2. SpectralSequences/sseq

Codex used code and designs from the `ext` implementation in the
SpectralSequences/sseq repository as structural and implementation references.

- Project: [SpectralSequences/sseq](https://github.com/SpectralSequences/sseq),
  especially its [`ext`](https://github.com/SpectralSequences/sseq/tree/master/ext)
  directory
- Authors named in the `ext` package metadata: Hood Chatham, Dexter Chua, and
  Joey Beauvais-Feisthauer
- License: MIT or Apache License 2.0

The local-to-upstream function mapping, including the relationship assigned to
each entry, is in [PROVENANCE.md](PROVENANCE.md). Other upstream contributors
retain attribution for their contributions. This distribution uses the Apache
License 2.0 option for material derived from SpectralSequences/sseq.

## 3. Nassau algorithm and cnassau/steenrod

The resolution algorithm is based on Christian Nassau's paper,
[“Computing a minimal resolution over the Steenrod algebra”](https://arxiv.org/abs/1910.04063).
Nassau's [cnassau/steenrod](https://github.com/cnassau/steenrod) implementation
is a related implementation licensed under GPL-2.0. It is cited for context,
but no source code from it is copied, translated, linked, or distributed in
this project. The GPL-2.0 license therefore does not apply to this
distribution. See [PROVENANCE.md](PROVENANCE.md) for the distinction between
the published mathematical algorithm and the code provenance of this project.
