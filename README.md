# ext

`ext` is research software that uses the Nassau algorithm to compute a minimal
free resolution of the trivial module `F2` over the mod-2 Steenrod algebra.
Coefficients are represented in the Milnor basis.

## Build

From the repository root:

```bash
cargo build --release --locked
```

The executable is:

```text
target/release/ext
```

## Quick start

Start a calculation:

```bash
./target/release/ext compute \
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
the end, to `checkpoint/t100.checkpoint` for the example above. The
`checkpoint` directory is created automatically. A later command with a larger
`--t` automatically resumes from the latest checkpoint in that directory that
does not exceed the requested degree, then saves a new `tN.checkpoint`. Use
`--no-checkpoint` for a run that should neither load nor save a checkpoint. Use
`--verbose` when detailed startup, cache, CPU, memory, checkpoint, and timing
diagnostics are needed.

## Save and resume a calculation

Checkpoints are directories, not single files. A checkpoint contains a
manifest plus packed generator and differential data:

```text
checkpoint/
  t100.checkpoint/
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
./target/release/ext compute \
  --t 140 \
  --checkpoint-every-layers 10
```

This saves after every 10 newly completed layers. Use
`--checkpoint-every-layers 1` to save after every layer. Each intermediate
checkpoint is named for the layer it actually contains: for example, the save
after completing `t=110` is `checkpoint/t110.checkpoint`. After the next
checkpoint has been fully written and verified, the previous intermediate
checkpoint created by the same process is removed. Thus a run does not leave a
checkpoint for every interval. The checkpoint loaded at startup is not removed,
and explicitly named checkpoint paths continue to be updated in place.

For example, after completing `--t 100`, extend the calculation with:

```bash
./target/release/ext compute \
  --t 120
```

This loads `checkpoint/t100.checkpoint` and saves the extended result as
`checkpoint/t120.checkpoint`; the `t100` checkpoint remains unchanged.

Use `--checkpoint DIR` to choose a different checkpoint path:

```bash
./target/release/ext compute \
  --t 100 \
  --checkpoint runs/local/resolution.checkpoint
```

The same path can then be resumed to a larger degree:

```bash
./target/release/ext compute \
  --t 120 \
  --checkpoint runs/local/resolution.checkpoint
```

Use `--fresh` to ignore an existing checkpoint and recompute from the
beginning. When extending or testing a calculation, prefer different input and
output paths:

```bash
./target/release/ext compute \
  --t 140 \
  --load-checkpoint runs/local/resolution.checkpoint \
  --save-checkpoint runs/local/t140.checkpoint
```

## Export results

Checkpoints are optimized for saving and resuming a calculation. They are not
the public data format. Two read-only export commands provide distinct views
of a completed checkpoint.

### Bidegree ranks

Export one row per nonzero `(s, t)` to CSV, with columns `s,t,rank`:

```bash
./target/release/ext export-bidegree-ranks \
  --t 140
```

This reads `checkpoint/t140.checkpoint` and writes `rank_output/t140.csv`,
creating `rank_output` automatically. With neither `--t` nor `--checkpoint`,
the latest default checkpoint is used.

For an explicitly named checkpoint or output file:

```bash
./target/release/ext export-bidegree-ranks \
  --checkpoint runs/local/t140.checkpoint \
  --output runs/local/basis_t140.csv
```

### Complete minimal resolution

Export every generator and differential to versioned JSON Lines:

```bash
./target/release/ext export-resolution \
  --t 140
```

This reads `checkpoint/t140.checkpoint` and writes
`resolution_output/t140.jsonl`, creating `resolution_output` automatically.
The checkpoint is only read; it is not rewritten.

The JSONL file is a readable exchange export, not a resumable checkpoint.
Because it stores JSON field names, generator names, and array structure, it
can be substantially larger than the packed binary checkpoint. Keep the binary
checkpoint for resuming a calculation, and generate JSONL when you need to
inspect or share the complete resolution.

The first line is a metadata record containing the format name, format
version, coefficient field, basis, range, record counts, and checkpoint
provenance. Every remaining line is one generator. A `t=100` export begins:

```json
{"record_type":"metadata","format":"ext-minimal-resolution-jsonl","format_version":1,"prime":2,"coefficient_field":"F2","algebra":"mod-2 Steenrod algebra","basis":"Milnor","max_internal_degree":100,"generator_count":1246,"differential_term_count":436836,"rank_convention":"rank(s,t) is the number of minimal generators in bidegree (s,t), equal to dim_F2 Ext_A^{s,t}(F2,F2)","source_checkpoint":{"format":"NMR_SHARDED_V1","format_version":1,"completed_internal_degree":100,"source_git_commit":"637757581d2b"}}
{"record_type":"generator","format_version":1,"id":0,"name":"g0","s":0,"t":0,"stem":0,"differential":[]}
{"record_type":"generator","format_version":1,"id":1,"name":"g1_1_1","s":1,"t":1,"stem":0,"differential":[{"target_id":0,"target_name":"g0","milnor":[1]}]}
```

Here `milnor: [4,1]` denotes the Milnor basis element `Sq(4,1)`. An empty
array denotes the unit. The `differential` array is summed over `F2`.
The name `g1_1_1` encodes `g{s}_{t}_{id}` with `s=1`, `t=1`, and global
generator ID `1`. Generator IDs and names are stable within the exported
checkpoint.

The v1 record definition is
[`schema/minimal-resolution-v1.schema.json`](schema/minimal-resolution-v1.schema.json).
Consumers should check the metadata `format` and `format_version` before
reading generator records.

For explicitly named paths:

```bash
./target/release/ext export-resolution \
  --checkpoint runs/local/t140.checkpoint \
  --output runs/local/minimal_resolution_t140.jsonl
```

Both export commands refuse to replace an existing output unless
`--overwrite` is supplied.

## Scope and limitations

- The implementation is for the prime `2` only.
- The command-line interface computes the resolution of the trivial module
  `F2`; it is not currently a general module-resolution library.
- Checkpoints should be treated as computational artifacts for this program.
  Use `export-resolution` when distributing complete minimal-resolution data.

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
