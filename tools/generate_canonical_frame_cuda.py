"""Generate guarded CUDA canonical-frame response multipliers."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from generativeqc_compiler.method.canonical_frame_response import (
    emit_canonical_frame_cuda,
)

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_if_changed(args.output, emit_canonical_frame_cuda())


if __name__ == "__main__":
    main()
