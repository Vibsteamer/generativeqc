"""Emit one FP64 shell operator for the molecular F12 integral library."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from generativeqc_compiler.integral.df_derivatives_cuda import (
    emit_df_derivatives_cpu,
    emit_df_derivatives_cuda,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("host", "cuda"), required=True)
    parser.add_argument("--domain", choices=("g", "h", "i"), required=True)
    parser.add_argument("--operator", choices=("coulomb", "geminal"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    emit = (
        emit_df_derivatives_cuda if args.backend == "cuda" else emit_df_derivatives_cpu
    )
    content = emit(
        auxiliary_h=args.domain != "i",
        maximum_orbital={"g": 4, "h": 5, "i": 6}[args.domain],
        fitted_geminal=args.operator == "geminal",
        cabs_h=args.domain == "h",
        cabs_i=args.domain == "i",
    )
    if args.output.is_file() and args.output.read_text(encoding="utf-8") == content:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
