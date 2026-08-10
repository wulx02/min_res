# Detailed subalgebra data

This directory contains self-generated subalgebra support tables bundled with
the project. They are required by the automatic subalgebra selector for the
corresponding cases.

The three published datasets and their metadata are embedded in the compiled
executable. These files remain in the repository so users can inspect their
provenance and verify their checksums. Advanced users can override an embedded
table with the corresponding `NASSAU_*_EXT_BITSET` and
`NASSAU_*_EXT_METADATA` environment variables.

Each dataset is published as:

- a `.bin` data file;
- a `.json` metadata file; and
- a `.sha256` checksum file.

To verify the files from the repository root, run:

```bash
for checksum in detailed_subalgebra/*.sha256; do
  (cd detailed_subalgebra && shasum -a 256 -c "$(basename "$checksum")")
done
```

These generated data files are distributed under the same MIT License as the
rest of the repository.
