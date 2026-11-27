"""Convert a local BSE JSON or NWChem basis file to GenerativeQC's checked schema.

Prints the basis identity, then one ``Z=<n> content_sha256=<hash>`` line per
element (see ``generativeqc.basis.element_content_sha256``).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from generativeqc.basis import element_content_sha256
from generativeqc.basis_import import import_bse, import_nwchem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--representation", choices=("cartesian", "spherical"))
    parser.add_argument(
        "--format",
        choices=("auto", "bse-json", "nwchem"),
        default="auto",
        help="auto treats .nw/.nwchem files as NWChem and anything else as BSE JSON",
    )
    parser.add_argument("--name", help="basis name (default: BSE name or file stem)")
    args = parser.parse_args()
    form = args.format
    if form == "auto":
        form = (
            "nwchem" if args.input.suffix.lower() in (".nw", ".nwchem") else "bse-json"
        )
    basis = (import_nwchem if form == "nwchem" else import_bse)(
        args.input,
        source=args.source,
        source_version=args.source_version,
        license=args.license,
        representation=args.representation,
        name=args.name,
    )
    basis.write(args.output)
    print(basis.identity)
    for element in basis.elements:
        content = element_content_sha256(element, basis.representation)
        print(f"Z={element.atomic_number} content_sha256={content}")


if __name__ == "__main__":
    main()
