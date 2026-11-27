# Decision: hash-pinned external methyl-halide F12 data

Status: implemented
Date: 2026-11-27

## Problem

The CH3Br/CH3I scalar-ECP F12 presets were package data. They contained
ccRepo Br/I aug-cc-pV{D,T}Z-PP-F12 orbital, OptRI and MP2Fit records, the
ccRepo H cc-pV{D,T}Z-F12 MP2Fit records, the published C aug-cc-pV{D,T}Z-F12
diffuse supplements and NIST CCCBDB geometries. None of these sources grants
redistribution. The project keeps only redistributable third-party data with
pinned provenance and notices in the repository.

## Decision

The package ships `python/generativeqc/data/f12_methyl_halide_external.json`
instead. It contains no exponents, coefficients or coordinates. For each user
file it records the source, request, retrieval date, file SHA-256, content
SHA-256 and citation. For each basis role it records how the record is
assembled from user files and from the installed Basis Set Exchange package,
the content SHA-256 of every element, and the identity of the assembled record.

`methyl_halide_basis`, `methyl_halide_geometry` and `methyl_halide_plan` read
the files from `data_dir` or `GENERATIVEQC_EXTERNAL_DATA_DIR`, in the
`f12_methyl_halide` subdirectory. Each file is read through
`read_local_bytes`, which refuses it above 64 MiB before parsing. A basis file
must hold exactly its recorded element; any other element in it is refused.
Element records are accepted by `element_content_sha256`, which covers binary64
shell and ECP values and the AO representation. A ccRepo download and a
re-spelled export of the same values are therefore both accepted. Missing
files, other values and a Basis Set
Exchange release with changed records are refused with the expected hash and
the acquisition procedure. There is no substitute basis. After assembly each
role identity and the `FocalBasis` `basis_id` must equal the recorded values,
so a successful load reproduces the previously packaged records exactly.

## Rejected alternatives

- Accepting files by byte hash alone. Harmless respelling of a record, such as
  a Basis Set Exchange NWChem export of the same values, would be refused. The
  byte hash is recorded to identify the documented download only.
- Falling back to redistributable Basis Set Exchange bases when files are
  missing. That silently changes the method and invalidates the bound results.
  Tests that need only a bromine ECP or a basis-admission check use explicit
  Basis Set Exchange records instead.
- Hashing the canonical assembled JSON only. A user could not tell which input
  file was wrong, and the hash could not be checked per element.

## Invariants

- The manifest never contains basis values or coordinates.
- Never substitute another basis set or geometry for a recorded input.
- A content hash covers one element record, not the rest of its file. The
  parsed element set must equal the recorded element before the hash is taken.
- Changing the assembly layout, grouping or ECP text changes the basis
  identity. That requires new identities and new validation.
- Geometry conversion uses the CODATA 2010 Bohr radius `0.52917721092` Å,
  which the recorded plans used.

## Evidence

The loader was run outside the test suite against locally retrieved ccRepo
files, the published diffuse supplements, the CCCBDB geometries and
`basis_set_exchange` 0.12. All 16 role identities and all four `basis_id`
values equal those of the formerly packaged records, and the Bohr geometries
are bit-identical. The 48 element content hashes include 27 distinct records.

`tests/python/test_f12_methyl_halide.py` checks the loader with synthetic
files and Basis Set Exchange records. Its external tier repeats the basis-size
and fitted-ECP checks with the acquired inputs. A file containing an added
element or one exceeding the pre-parse size limit is refused before its basis
can enter a calculation.

## Consequences

Users must obtain the files themselves. Basis Set Exchange upgrades that alter
any used record are refused until the hashes are re-verified and the verified
version is updated in the manifest.

## Revisit when

- ccRepo grants redistribution, or Basis Set Exchange publishes records with
  the same content hashes.
- A decision replaces the H MP2Fit record. That changes the `ri` identities and
  every `basis_id`, so the bound results no longer apply.

## References

- `docs/user/restricted_reference_data.md`
- `tests/python/test_f12_methyl_halide.py`
