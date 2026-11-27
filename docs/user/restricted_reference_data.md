# Obtain restricted basis sets and reference data

Some calculations use basis sets or geometries whose publishers do not permit
redistribution. GenerativeQC does not ship them. It records the SHA-256 of each
file and of each element record, reads your own copies from a local directory
and refuses anything else. It never substitutes a different basis set or
geometry.

## Where the files go

Put the files in the `f12_methyl_halide` subdirectory of one directory. Pass
that directory as `data_dir=` or set `GENERATIVEQC_EXTERNAL_DATA_DIR`:

```bash
export GENERATIVEQC_EXTERNAL_DATA_DIR="$HOME/generativeqc-external"
mkdir -p "$GENERATIVEQC_EXTERNAL_DATA_DIR/f12_methyl_halide"
```

Then check every selection:

```python
from generativeqc.experimental.f12_methyl_halide import (
    methyl_halide_basis,
    methyl_halide_geometry,
    required_files,
)

for halogen in ("Br", "I"):
    print(halogen, required_files(halogen))
    methyl_halide_geometry(halogen)
    for cardinal in (2, 3):
        print(halogen, cardinal, methyl_halide_basis(halogen, cardinal).basis_id)
```

A missing file raises `FileNotFoundError`, which lists every missing file and
how to obtain it. A file holding other values raises `ValueError` with the
expected and actual content SHA-256. A basis file that also holds records for an
element other than its listed one raises `ValueError`, and so does any file
larger than 64 MiB, before it is parsed.

## CH3Br and CH3I scalar-ECP F12 inputs

`methyl_halide_basis(halogen, cardinal)` assembles the aug-cc-pV{D,T}Z-PP-F12
selection for CH3Br or CH3I: orbital, def2-universal-JKFIT, element MP2Fit and
OptRI CABS records. `methyl_halide_plan` uses it with the geometry from
`methyl_halide_geometry`. Records published by the Basis Set Exchange are read
from the installed `basis_set_exchange` package
(`pip install 'generativeqc[bse]'`). The verified release is 0.12; a release
with different values for these records is refused. The files below supply the
rest.

| Source | Content | Cite |
| --- | --- | --- |
| [ccRepo](http://www.grant-hill.group.shef.ac.uk/ccrepo/) (J. G. Hill) | Br and I aug-cc-pV{D,T}Z-PP-F12 orbital, OptRI and MP2Fit; H cc-pV{D,T}Z-F12 MP2Fit | Br, I: J. G. Hill and K. A. Peterson, J. Chem. Phys. 141, 094106 (2014); J. G. Hill, J. Chem. Phys. 160, 114102 (2024), doi:10.1063/5.0197210. H: S. Kritikou and J. G. Hill, J. Chem. Theory Comput. 11, 5269 (2015), doi:10.1021/acs.jctc.5b00816 |
| Sylvetsky, Kesharwani and Martin, [Table I](https://doi.org/10.1063/1.4998332) | Carbon diffuse functions of aug-cc-pV{D,T}Z-F12 | J. Chem. Phys. 147, 134106 (2017) |
| [NIST CCCBDB](https://cccbdb.nist.gov/) | Experimental CH3Br and CH3I geometries | NIST Standard Reference Database 101; K. Kuchitsu (ed.), Structure of Free Polyatomic Molecules – Basic Data (1998) |

### ccRepo records

ccRepo publishes no version identifiers, so the content hash identifies each
record. The Br and I pages take an HTTP POST form and the H page an HTTP GET
query, each with the fields `basis` and `program=NWChem`. The MP2Fit and OptRI
names do not appear in the Br and I web forms; submit the field values
directly. The NWChem text is the first `<nobr>` element of the response. This
script writes it in the recorded form:

```python
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

CCREPO = "http://www.grant-hill.group.shef.ac.uk/ccrepo"
PAGES = {
    "Br": ("bromine/brbasis.php", True),
    "I": ("iodine/ibasis.php", True),
    "H": ("hydrogen/hbasis.php", False),
}


class FirstNobr(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth, self.done, self.parts = 0, False, []

    def handle_starttag(self, tag, attrs):
        if tag == "nobr" and not self.done:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag == "nobr" and self.depth:
            self.depth -= 1
            self.done = self.depth == 0

    def handle_data(self, data):
        if self.depth and not self.done:
            self.parts.append(data)


def fetch(element, basis, folder):
    page, post = PAGES[element]
    fields = urllib.parse.urlencode({"basis": basis, "program": "NWChem"})
    url = f"{CCREPO}/{page}"
    request = (
        urllib.request.Request(url, data=fields.encode())
        if post
        else urllib.request.Request(f"{url}?{fields}")
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        parser = FirstNobr()
        parser.feed(response.read().decode("utf-8", "replace"))
    lines = [re.sub(r"\s+", " ", x).strip() for x in "\n".join(parser.parts).splitlines()]
    path = Path(folder) / f"{element}_{basis}.nwchem"
    path.write_bytes(("\n".join(x for x in lines if x) + "\n").encode())


folder = Path.home() / "generativeqc-external/f12_methyl_halide"
for element in ("Br", "I"):
    for n in "DT":
        for suffix in ("", "_OPTRI", "_MP2FIT"):
            fetch(element, f"aug-cc-pV{n}Z-PP-F12{suffix}", folder)
for n in "DT":
    fetch("H", f"cc-pV{n}Z-F12_MP2FIT", folder)
```

### Carbon diffuse functions

Write the carbon exponents of the aug-cc-pV{n}Z-F12 row of Table I as an NWChem
block: a `BASIS "ao basis" SPHERICAL PRINT` line, one uncontracted shell per
exponent with coefficient 1.0 in increasing angular momentum, then `END`. The
double-zeta file has one d shell and the triple-zeta file one d and one f shell.

### Geometries

Open the CCCBDB experimental geometry page in the table below and write the
listed Cartesian coordinates in Å as an XYZ file. Order the atoms C, halogen,
H, H, H, keeping the listed order of the hydrogens. The comment line is free.

### Files and hashes

| File | Source | Request | Retrieved |
| --- | --- | --- | --- |
| `Br_aug-cc-pVDZ-PP-F12.nwchem` | ccRepo Br | `basis=aug-cc-pVDZ-PP-F12` | 2026-11-17 |
| `Br_aug-cc-pVDZ-PP-F12_MP2FIT.nwchem` | ccRepo Br | `basis=aug-cc-pVDZ-PP-F12_MP2FIT` | 2026-11-20 |
| `Br_aug-cc-pVDZ-PP-F12_OPTRI.nwchem` | ccRepo Br | `basis=aug-cc-pVDZ-PP-F12_OPTRI` | 2026-11-20 |
| `Br_aug-cc-pVTZ-PP-F12.nwchem` | ccRepo Br | `basis=aug-cc-pVTZ-PP-F12` | 2026-11-17 |
| `Br_aug-cc-pVTZ-PP-F12_MP2FIT.nwchem` | ccRepo Br | `basis=aug-cc-pVTZ-PP-F12_MP2FIT` | 2026-11-18 |
| `Br_aug-cc-pVTZ-PP-F12_OPTRI.nwchem` | ccRepo Br | `basis=aug-cc-pVTZ-PP-F12_OPTRI` | 2026-11-18 |
| `CH3Br.xyz` | NIST CCCBDB | [CAS 74-83-9](https://cccbdb.nist.gov/expgeom2x.asp?casno=74839) | 2026-11-21 |
| `CH3I.xyz` | NIST CCCBDB | [CAS 74-88-4](https://cccbdb.nist.gov/expgeom2x.asp?casno=74884) | 2026-11-21 |
| `C_aug-cc-pVDZ-F12_diffuse.nwchem` | Sylvetsky et al. | Table I | published 2017 |
| `C_aug-cc-pVTZ-F12_diffuse.nwchem` | Sylvetsky et al. | Table I | published 2017 |
| `H_cc-pVDZ-F12_MP2FIT.nwchem` | ccRepo H | `basis=cc-pVDZ-F12_MP2FIT` | 2026-11-20 |
| `H_cc-pVTZ-F12_MP2FIT.nwchem` | ccRepo H | `basis=cc-pVTZ-F12_MP2FIT` | 2026-11-19 |
| `I_aug-cc-pVDZ-PP-F12.nwchem` | ccRepo I | `basis=aug-cc-pVDZ-PP-F12` | 2026-11-17 |
| `I_aug-cc-pVDZ-PP-F12_MP2FIT.nwchem` | ccRepo I | `basis=aug-cc-pVDZ-PP-F12_MP2FIT` | 2026-11-20 |
| `I_aug-cc-pVDZ-PP-F12_OPTRI.nwchem` | ccRepo I | `basis=aug-cc-pVDZ-PP-F12_OPTRI` | 2026-11-20 |
| `I_aug-cc-pVTZ-PP-F12.nwchem` | ccRepo I | `basis=aug-cc-pVTZ-PP-F12` | 2026-11-17 |
| `I_aug-cc-pVTZ-PP-F12_MP2FIT.nwchem` | ccRepo I | `basis=aug-cc-pVTZ-PP-F12_MP2FIT` | 2026-11-18 |
| `I_aug-cc-pVTZ-PP-F12_OPTRI.nwchem` | ccRepo I | `basis=aug-cc-pVTZ-PP-F12_OPTRI` | 2026-11-18 |

The loader accepts a basis file when it holds only the listed element and that
element's content SHA-256 matches. The file SHA-256 identifies the bytes the
procedures above produce; a file with other spelling or formatting of the same
values is also accepted.

| File | File SHA-256 | Content SHA-256 |
| --- | --- | --- |
| `Br_aug-cc-pVDZ-PP-F12.nwchem` | `c0f9cf0f3404279d7a9f28706f713c2ed0e788437341cb83dfe0d5ce5dba39f2` | `8aa43efc27e243947bc303da8568de89d9d95e392b403f7d36c94e7bc3fdc02f` |
| `Br_aug-cc-pVDZ-PP-F12_MP2FIT.nwchem` | `1a96c964998f15d74e9d88008035aff30b81f0bc30c509882232c23597f06fae` | `afbc10c2cd798c9c49a98752ecc43b6081602568afe5f0d866e9fc993237bcdb` |
| `Br_aug-cc-pVDZ-PP-F12_OPTRI.nwchem` | `74037f506913138bc892a85359228c944bd94ad6a290a27b1c1469354806dd30` | `6154a7f094e0c1c7525eca6e00bd793f73373b339d185a17be942bfce05489c6` |
| `Br_aug-cc-pVTZ-PP-F12.nwchem` | `4b3e69582cbdbc8bd23a63c2c0c9c69bf2c708219cbece088840ac998786e274` | `417f2b8f2a68a764dd41fdad273ecc1aa1c104afb1f5a9632acfea8371a4a9e4` |
| `Br_aug-cc-pVTZ-PP-F12_MP2FIT.nwchem` | `3d520522768e068dda12b90c9b959a8bb0bdb962632e3a5922fc0911faa527c8` | `3f8e82d2e5c74c6570a7ef70b91d6e50a70f119b4c78039249396e485b9069cd` |
| `Br_aug-cc-pVTZ-PP-F12_OPTRI.nwchem` | `42a39e6817e60e3729b8431cd4b83dcc014f142e369b2a534e01555180b8f539` | `c8f9cfe636285c4c9dceb533800b8f732a9815c9544fad0b4324287ceaf5f7d6` |
| `CH3Br.xyz` | `cba0117c67c463558e378c44498bb0a8f120f41cfd2ffee3de49ecccffa639fe` | `4c6f9f5a5cbdcd3dafb88ed048391645cb2b1afa1d3db65e767e0dbadea79219` |
| `CH3I.xyz` | `3d8fbad99d26da979d30e608848077d788d458dd2578dd5b003cedc881cee708` | `4bb0bf265ab9373c7f2e532bcf59350dce3bb350bab7ff15e441a94ac23b9346` |
| `C_aug-cc-pVDZ-F12_diffuse.nwchem` | `e814596f8225848a05ed8498492ddce44facf404f5ea3d91c656d50fb736efda` | `d4b5542153cec91baf709dfc381054730e6550ce044ed7ab35a4f17c3659f9a2` |
| `C_aug-cc-pVTZ-F12_diffuse.nwchem` | `908c800a9985de19ae9ec64895ce3d110320a9d6ac50c05172bf008ad116435f` | `00bdf917b4fdfdc27af02f7d19807a63414a06aa277b4b0aa5e932e2b43c9ced` |
| `H_cc-pVDZ-F12_MP2FIT.nwchem` | `d711cff115eb7ffe477088e142f523b986c6f70407c6b2832f2f3251b257aec1` | `ff2198eea86f947419584f9cf3c08b2bdceb65203a0deb8dff4b9c214723d41d` |
| `H_cc-pVTZ-F12_MP2FIT.nwchem` | `ca0adf154be0ff78185827148821218780a3d87df14ff32e240bc12e0b133101` | `b903db67ddd17eb4e82b89c83f50f21d927745330a3f77df576fc0cc31035542` |
| `I_aug-cc-pVDZ-PP-F12.nwchem` | `27a44f354e5e6dc1ac1295354ca8c50b3a52cf372a7144631dc9e66958938fd0` | `78bc22318cad55503b33cfa34a7ceda66fbeeb98cf5f8163986c7f42fbf6ca46` |
| `I_aug-cc-pVDZ-PP-F12_MP2FIT.nwchem` | `9987466366f49030417520e300482173e4b9a3c8b4b1c7cc4879835c76100547` | `d21b2b477b5adcc76b777afa29803ef63abf2b7879ce179c4db5316eb636a90c` |
| `I_aug-cc-pVDZ-PP-F12_OPTRI.nwchem` | `7fb16c32138c8a3b5e5dd3f48547524c8de9f77b2101370ddc362a842b2aa757` | `09e6496fcc1045539f9aefff0d6a920f0317545f4d55d6b7398df39bc58e0837` |
| `I_aug-cc-pVTZ-PP-F12.nwchem` | `297a93fa530becddbe7442860824253efbc47fa9ceb3e4ef2cae931365725443` | `a49fd743359484c051f9aaefa46f4bcaf37cdc62c2e01adff24027af2cef423e` |
| `I_aug-cc-pVTZ-PP-F12_MP2FIT.nwchem` | `5e0b9e9a4e09f92e512e12f8f8e5fe4bbfd52ef2838d91743126675a80bbeaaf` | `0b2ac92ef18e6c8e8aa187d7f166b31f60e3d748a2ea4cdf4df9fc68233a4d3f` |
| `I_aug-cc-pVTZ-PP-F12_OPTRI.nwchem` | `d223a16e3f9aaacd7738c8f610a20bc17b4c5d89ae3f6d297310251739cd105f` | `0eabf47144fdabdcef307c2d4c57cb358b611f326676af073cf836f1259a053d` |

### Inspect or convert a file

`tools/import_basis.py` reads the NWChem files and prints the content SHA-256 of
each element, for comparison with the table:

```bash
python tools/import_basis.py \
  "$GENERATIVEQC_EXTERNAL_DATA_DIR/f12_methyl_halide/Br_aug-cc-pVDZ-PP-F12.nwchem" \
  br-aug-cc-pvdz-pp-f12.json --format nwchem \
  --source 'ccRepo, http://www.grant-hill.group.shef.ac.uk/ccrepo/' \
  --source-version 'retrieved <date>' --license 'not redistributable' \
  --representation spherical
```

`python/generativeqc/data/f12_methyl_halide_external.json` holds the same
records in machine-readable form, together with the assembly of each basis role
and the identities of the assembled records. See
[local basis data](external_basis.md) for the import formats and the content
hash.
