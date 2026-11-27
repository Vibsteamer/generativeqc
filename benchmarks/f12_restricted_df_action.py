"""Conventional correlation-DF part of the restricted fixed-geminal CC equations."""

from collections.abc import Callable, Iterator
from functools import lru_cache
from string import ascii_lowercase

import numpy as np
from generativeqc_compiler.cc.df_equations import build_df_virtual_correction_program
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.cc.df_source import (
    BLOCK_FACTORS,
    PHYSICAL_FACTORS,
    factor_embedding_vjp,
    factor_program,
    retained_factor_program,
)
from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.tensor import Program, einsum, transpose_program


@lru_cache(maxsize=16)
def _q_reduction(o: int, v: int, batch_size: int) -> Program:
    row = build_df_virtual_correction_program(o, v)
    batch = matrix_program(row, batch_size=batch_size)
    outputs = {}
    for name, node in batch.outputs.items():
        word = ascii_lowercase[: len(node.spec.indices)]
        outputs[name] = einsum(word + "->" + word[1:], node)
    return Program(outputs, provenance=batch.provenance)


def _only(program: Program, inputs: dict) -> dict:
    return {
        node.attrs["name"]: inputs[node.attrs["name"]]
        for node in program.live_nodes
        if node.op == "input"
    }


class RestrictedDFAction:
    """Evaluate E/R and their adjoints from a symmetric whitened B[Q,p,q] using bounded Q batches.

    Factor symmetry is projected in the input map. Nuclear and raw-metric derivatives
    belong upstream. All graph execution is delegated to the same supplied backend.
    Each reverse invocation uses the exact same amplitudes, physical factors and
    reduction boundaries as its primal. The final short Q batch has its own shape.
    """

    def __init__(
        self,
        f: np.ndarray,
        bmo: np.ndarray,
        occupied: int,
        run: Callable,
        *,
        batch_size: int = 8,
    ) -> None:
        o = occupied
        if (
            f.ndim != 2
            or f.shape[0] != f.shape[1]
            or bmo.ndim != 3
            or bmo.shape[1:] != f.shape
        ):
            raise ValueError(
                "square Fock and Q-major full spatial factors are required"
            )
        n = len(f)
        q = len(bmo)
        v = n - o
        if (
            type(o) is not int
            or min(o, v, q) < 1
            or type(batch_size) is not int
            or batch_size < 1
        ):
            raise ValueError("invalid occupied, virtual or auxiliary batch dimensions")
        self.adjoints = {}
        self.run = run
        self.o = o
        self.v = v
        self.q = q
        self.batch_size = batch_size
        sectors = factor_program(o, v, q, symmetric_pairs=True)
        self.sectors = run(
            Program({k: sectors.outputs[k] for k in PHYSICAL_FACTORS}),
            {"bmo": np.moveaxis(bmo, 0, -1)},
        ).outputs
        self.retained = retained_factor_program(o, v, q)
        retained = Program({k: self.retained.outputs[k] for k in BLOCK_FACTORS})
        self.blocks = run(retained, self.sectors).outputs
        self.h = {**self.blocks, "foo": f[:o, :o], "fvv": f[o:, o:], "fov": f[:o, o:]}
        self.core = build_ccsd_program(
            o, v, form="optimized", diagnostics=False, external_virtual_correction=True
        )

    def _reverse(self, program: Program, fields: tuple[str, ...]) -> Program:
        key = (program.logical_hash, fields)
        if key not in self.adjoints:
            self.adjoints[key] = transpose_program(
                program, tuple(program.outputs), inputs=fields
            ).program
        return self.adjoints[key]

    def _batches(self) -> Iterator[tuple[slice, Program, dict]]:
        for start in range(0, self.q, self.batch_size):
            section = slice(start, min(start + self.batch_size, self.q))
            graph = _q_reduction(self.o, self.v, section.stop - start)
            yield section, graph, {k: self.sectors[k][section] for k in ("bov", "bvv")}

    def evaluate(self, t1: np.ndarray, t2: np.ndarray) -> dict:
        amplitudes = {"t1": t1, "t2": t2}
        virtual = {}
        for _, graph, factors in self._batches():
            row = self.run(graph, {**amplitudes, **factors}).outputs
            if not virtual:
                virtual = {k: x.copy() for k, x in row.items()}
            else:
                for k, x in row.items():
                    virtual[k] += x
        return self.run(
            self.core, _only(self.core, {**self.h, **amplitudes, **virtual})
        ).outputs

    def adjoint(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        l1: np.ndarray,
        l2: np.ndarray,
        *,
        amplitudes: bool,
    ) -> dict:
        """Gradient of E + l1:R1 + l2:R2 at fixed t1/t2 and Lambda.

        The ordinary residual is additive and linear in the virtual correction.
        Its transpose broadcasts the residual cotangents to each Q tile. Input
        and output projections remain the audited restricted spatial ones.
        """
        t = {"t1": t1, "t2": t2}
        corrections = ("df_virtual_singles", "df_virtual_doubles")
        sites = tuple(t) if amplitudes else tuple(self.h)
        core = self._reverse(self.core, sites + corrections)
        seed = {
            "bar_correlation_energy": np.array(1.0),
            "bar_singles_residual": l1,
            "bar_doubles_residual": l2,
        }
        primary = self.run(core, _only(core, {**self.h, **t, **seed})).outputs
        qseeds = {f"bar_{k}": primary[f"bar_{k}"] for k in corrections}
        result = {k: primary["bar_" + k].copy() for k in sites}
        qbars = (
            {} if amplitudes else {k: np.zeros_like(y) for k, y in self.sectors.items()}
        )
        for section, graph, factors in self._batches():
            vjp = self._reverse(graph, tuple(t) if amplitudes else ("bov", "bvv"))
            qb = self.run(vjp, _only(vjp, {**t, **factors, **qseeds})).outputs
            for bar, val in qb.items():
                k = bar.removeprefix("bar_")
                if amplitudes:
                    result[k] += val
                else:
                    qbars[k][section] += val
        if amplitudes:
            return result
        retained = Program({k: self.retained.outputs[k] for k in BLOCK_FACTORS})
        r = self._reverse(retained, PHYSICAL_FACTORS)
        bars = self.run(
            r,
            _only(
                r, {**self.sectors, **{"bar_" + k: result[k] for k in BLOCK_FACTORS}}
            ),
        ).outputs
        embed = factor_embedding_vjp(self.o, self.v, self.q)
        bmo = self.run(
            embed, {"bar_" + k: qbars[k] + bars["bar_" + k] for k in PHYSICAL_FACTORS}
        ).outputs["bar_bmo"]
        f = np.zeros((self.o + self.v, self.o + self.v), dtype=t1.dtype)
        f[: self.o, : self.o] = result["foo"]
        f[: self.o, self.o :] = result["fov"]
        f[self.o :, self.o :] = result["fvv"]
        return {"f": f, "bmo": np.moveaxis(bmo, -1, 0)}
