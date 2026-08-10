# nassau_min_res

`nassau_min_res` is research software for computing a minimal
free resolution of the trivial module `F2` over the mod-2 Steenrod algebra.
Coefficients are represented in the Milnor basis.

## Build

From the repository root:

```bash
cargo build --release --locked
```

The executable is:

```text
target/release/nassau_min_res
```

## Quick start

Start a calculation:

```bash
./target/release/nassau_min_res compute \
  --t 100
```

`--t N` is the only range parameter. For every internal-degree layer
`1 <= t <= N`, the program computes the full task range `0 <= s < t`, which
gives the complete triangular output range `0 <= s <= t`.

No thread or checkpoint options are required. By default, Rayon uses the
logical CPUs available to the process. The fixed-`t` batch algorithm uses half
that thread count for its outer worker groups, with a minimum of one and a
maximum of four. Override these choices with `--threads N` or
`--batch-workers N` when needed.

Console output is concise by default: one progress line per completed `t`
layer and a short final summary. The completed calculation is saved once, at
the end, to `nassau_min_res.checkpoint`. A later command automatically resumes
from that checkpoint when extending the range. Use `--no-checkpoint` for a run
that should neither load nor save it. Use `--verbose` when detailed startup,
cache, CPU, memory, checkpoint, and timing diagnostics are needed.

## Save and resume a calculation

Checkpoints are directories, not single files. A checkpoint contains a
manifest plus packed generator and differential data:

```text
resolution.checkpoint/
  manifest.json
  gen_meta.pack
  diff_terms.pack
```

Checkpoint updates are written and verified in a temporary sibling directory
before the destination is atomically exchanged on macOS or Linux. The old tree
is then removed rather than retained as a backup. For safety, the program
refuses to replace a directory unless it is a valid checkpoint
containing only the three files shown above.

The default command saves only after the complete requested range succeeds.
To save intermediate progress as well, specify a number of completed `t`
layers. The final checkpoint is always written:

```bash
./target/release/nassau_min_res compute \
  --t 140 \
  --checkpoint-every-layers 10
```

This saves after every 10 newly completed layers. Use
`--checkpoint-every-layers 1` to save after every layer.

Use `--checkpoint DIR` to choose a different checkpoint path:

```bash
mkdir -p runs/local

./target/release/nassau_min_res compute \
  --t 100 \
  --checkpoint runs/local/resolution.checkpoint
```

The same path can then be resumed to a larger degree:

```bash
./target/release/nassau_min_res compute \
  --t 120 \
  --checkpoint runs/local/resolution.checkpoint
```

Use `--fresh` to ignore an existing checkpoint and recompute from the
beginning. When extending or testing a calculation, prefer different input and
output paths:

```bash
./target/release/nassau_min_res compute \
  --t 140 \
  --load-checkpoint runs/local/resolution.checkpoint \
  --save-checkpoint runs/local/t140.checkpoint
```

## Export results

Export one row per nonzero `(s, t)` to CSV, with columns `s,t,rank`:

```bash
./target/release/nassau_min_res export \
  --checkpoint runs/local/t140.checkpoint \
  --output runs/local/basis_t140.csv
```

To write a human-readable generator report directly from a calculation, use
`--output FILE`. Add `--show-differentials` when the differentials are also
needed. Pass an explicit `--checkpoint` if you do not want the checkpoint path
to be derived from the report filename.

```bash
./target/release/nassau_min_res compute \
  --t 120 \
  --checkpoint runs/local/t120.checkpoint \
  --output runs/local/resolution_t120.txt
```

## Scope and limitations

- The implementation is for the prime `2` only.
- The command-line interface computes the resolution of the trivial module
  `F2`; it is not currently a general module-resolution library.
- Checkpoints should be treated as versioned computational artifacts. Keep the
  program version and command metadata with any checkpoint you distribute.

## Development checks

Before submitting a change, run:

```bash
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo build --release --locked
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
