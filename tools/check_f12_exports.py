"""Check a focal shared library's public C functions against its ELF export map."""

import argparse
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("--nm", default="nm", help="nm for the target toolchain")
    args = parser.parse_args()
    # The build passes the same map to the linker. Require literal C names so
    # an accidental wildcard cannot silently broaden this ABI.
    text = args.manifest.read_text(encoding="utf-8")
    section = re.fullmatch(
        r"\s*\{\s*global:\s*((?:[A-Za-z_]\w*\s*;\s*)+)local:\s*\*\s*;\s*\}\s*;\s*",
        text,
    )
    if section is None:
        parser.error(f"invalid focal export map: {args.manifest}")
    expected = set(re.findall(r"[A-Za-z_]\w*", section[1]))
    listing = subprocess.check_output(
        [args.nm, "-D", "--defined-only", "--format=posix", str(args.library)],
        text=True,
    )
    exported = {}
    for line in listing.splitlines():
        name, kind, *_ = line.split()
        exported[name] = kind
    problems = []
    for name in sorted(expected - exported.keys()):
        problems.append(f"missing {name}")
    for name in sorted(exported.keys() - expected):
        problems.append(f"unexpected {name}")
    for name in sorted(expected & exported.keys()):
        if exported[name] != "T":
            problems.append(f"{name} has symbol type {exported[name]}, expected T")
    if problems:
        raise SystemExit(f"{args.library}:\n" + "\n".join(problems))
    print(f"{args.library}: verified {len(expected)} public C functions")


if __name__ == "__main__":
    main()
