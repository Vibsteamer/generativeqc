"""Build the molecular integral-adjoint qualification library for CUDA or CPU reference."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
from generativeqc_compiler.common.compiler_cache import (
    resolve_compiler_cache,
    run_cached_compiler,
)
from generativeqc_compiler.integral.df_derivatives_cuda import (
    emit_df_derivatives_cpu,
    emit_df_derivatives_cuda,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    parser.add_argument("--cxx", default=shutil.which("c++"))
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--backend", choices=["host", "cuda"], default="cuda")
    shell_domain = parser.add_mutually_exclusive_group()
    shell_domain.add_argument(
        "--cabs-h",
        action="store_true",
        help="Build a separately named FP64 variant for combined-basis h/h/h sources.",
    )
    shell_domain.add_argument(
        "--cabs-i",
        action="store_true",
        help="Build a separately named FP64 variant for combined-basis i/i/i sources.",
    )
    args = parser.parse_args()
    cuda = args.backend == "cuda"
    compiler = args.nvcc if cuda else args.cxx
    if not compiler or not args.cxx:
        parser.error(
            "the selected backend compiler and a host C++ compiler are required"
        )
    launcher = resolve_compiler_cache()
    out = args.output_dir.resolve()
    domain_suffix = "_cabs_i" if args.cabs_i else "_cabs_h" if args.cabs_h else ""
    build = out / (args.backend + domain_suffix)
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CCACHE_BASEDIR", str(REPO))
    stats = [str(launcher.path), "--show-stats"]
    if launcher.name == "ccache":
        (build / "cache-before.txt").write_text(
            subprocess.check_output(stats, text=True)
        )
    source = REPO / "benchmarks/f12_molecular_integral_sink.cu"
    common = [
        compiler,
        "-std=c++17",
        "-O2",
        "-I" + str(build),
        "-I" + str(REPO / "src"),
    ]
    if cuda:
        arch = args.arch.replace("sm_", "").replace("sm", "")
        if not arch.isdigit():
            parser.error("--arch must be an SM number such as sm_89")
        common += ["-arch=sm_" + arch, "-Xcompiler=-fPIC"]
    else:
        common += ["-x", "c++", "-fPIC", "-DGQC_SINK_HOST_REFERENCE"]
    if args.cabs_i:
        common.append("-DGQC_F12_CABS_I")
    elif args.cabs_h:
        common.append("-DGQC_F12_CABS_H")
    emit = emit_df_derivatives_cuda if cuda else emit_df_derivatives_cpu
    objects = []

    def run(argv: list[str], name: str) -> None:
        print(launcher.path, *argv, flush=True)
        ret = run_cached_compiler(
            argv, 1800, label="molecular integral sink", launcher=launcher
        )
        (build / (name + ".log")).write_text(ret.stdout + ret.stderr)
        if ret.returncode:
            sys.exit(ret.stderr or f"{name} exited {ret.returncode}")

    for operator in ("coulomb", "geminal"):
        header = build / ("molecular-" + operator + ".hpp")
        content = emit(
            auxiliary_h=not args.cabs_i,
            maximum_orbital=6 if args.cabs_i else 5 if args.cabs_h else 4,
            fitted_geminal=operator == "geminal",
            cabs_h=args.cabs_h,
            cabs_i=args.cabs_i,
        )
        if not header.exists() or header.read_text() != content:
            header.write_text(content)
        obj = build / (operator + ".o")
        flags = ["-DGQC_SINK_COULOMB"] if operator == "coulomb" else []
        run(common + flags + ["-c", str(source), "-o", str(obj)], operator)
        objects.append(str(obj))
    for source in (
        REPO / "src/molecule/basis.cpp",
        REPO / "benchmarks/f12_shell_projection.cpp",
    ):
        obj = build / (source.stem + ".o")
        run(
            [
                args.cxx,
                "-std=c++20",
                "-O2",
                "-fPIC",
                *(["-fvisibility=hidden"] if source.stem == "basis" else []),
                "-I" + str(REPO / "src"),
                "-I" + str(REPO / "include"),
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            source.stem,
        )
        objects.append(str(obj))
    suffix = "" if cuda else "_host"
    target = out / (
        "libgqc_f12_molecular_integral_sink" + domain_suffix + suffix + ".so"
    )
    # A failed relink must leave the last ready library intact.
    temporary_target = target.with_suffix(".linking.so")
    link_flags = ["-arch=sm_" + arch] if cuda else []
    run(
        [compiler, "-shared", *link_flags, *objects, "-o", str(temporary_target)],
        "link",
    )
    os.replace(temporary_target, target)
    if launcher.name == "ccache":
        (build / "cache-after.txt").write_text(
            subprocess.check_output(stats, text=True)
        )
    print(
        json.dumps(
            {"library": str(target), "cache": launcher.version, "backend": args.backend}
        )
    )


if __name__ == "__main__":
    main()
