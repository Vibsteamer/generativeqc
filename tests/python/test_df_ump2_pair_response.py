"""Full moving-frame spin-channel force against independently recanonicalized points."""
import numpy as np
import pytest
from generativeqc_compiler.mp2.unrestricted_df_response import df_ump2_pair_response
from generativeqc_compiler.tensor import execute


@pytest.mark.parametrize("channel", ["alpha_alpha", "alpha_beta", "beta_beta"])
def test_two_perturbations_of_whitened_factors_and_full_fock(channel: str) -> None:
    rng = np.random.default_rng(6418)
    dims = {"alpha": (3, 4), "beta": (2, 5)}
    left, right = channel.split("_")
    fields = {}
    for spin in dict.fromkeys((left, right)):
        no, nv = dims[spin]
        fields["B_" + spin] = rng.normal(size=(6, no, nv)) * 0.1
        fields["dB_" + spin] = rng.normal(size=(2, 6, no, nv)) * 0.08
        fields["eo_" + spin] = -np.linspace(1.4, 0.8, no)
        fields["ev_" + spin] = np.linspace(0.3, 1.3, nv)
        for name, size in (("dFo_", no), ("dFv_", nv)):
            d = rng.normal(size=(2, size, size)) * 0.2
            fields[name + spin] = d + d.transpose(0, 2, 1)
    graph = df_ump2_pair_response(channel, 6, *dims[left], *dims[right], 2)
    raw = execute(graph, fields).outputs

    def energy(x: int, step: float, vary_factor: bool, vary_fock: bool) -> float:
        factors, gaps = [], []
        for spin in (left, right):
            g = fields["B_" + spin] + (step if vary_factor else 0.0) * fields["dB_" + spin][x]
            fo = np.diag(fields["eo_" + spin]) + (step if vary_fock else 0.0) * fields["dFo_" + spin][x]
            fv = np.diag(fields["ev_" + spin]) + (step if vary_fock else 0.0) * fields["dFv_" + spin][x]
            eo, co = np.linalg.eigh(fo)
            ev, cv = np.linalg.eigh(fv)
            factors.append(co.T @ g @ cv)
            gaps.append(eo[:, None] - ev)
        v = np.einsum("Qia,Qjb->ijab", *factors)
        if left == right:
            v = v - v.swapaxes(2, 3)
        return float(np.sum(v**2 / (gaps[0][:, None, :, None] + gaps[1][None, :, None, :])) * (0.25 if left == right else 1.0))

    assert abs(raw["energy"] - energy(0, 0.0, False, False)) < 2e-12
    for x in range(2):
        for field, bf, ff in (("factor_force", True, False), ("fock_force", False, True), ("force", True, True)):
            step = 1e-5
            slope = (energy(x, -step, bf, ff) - energy(x, step, bf, ff)) / (2 * step)
            assert abs(raw[field][x] - slope) < 2e-10


def test_same_spin_dimension_identity() -> None:
    with pytest.raises(ValueError, match="same physical orbital space"):
        df_ump2_pair_response("beta_beta", 3, 2, 5, 2, 4, 1)
