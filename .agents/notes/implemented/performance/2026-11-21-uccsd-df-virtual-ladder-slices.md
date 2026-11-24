# Bound the bare DF virtual ladder in UCCSD

Implemented compiler term, 2026-11-21.

The all-active CH₃Br/ha-T case has 17 occupied and 185 virtual orbitals per spin. Its dense primal graph plans about 211 GB of device storage. The four live four-virtual integral inputs alone total 37.5 GB. The archived capacity study found that allocator changes preserved the dense storage bottleneck.

`build_uccsd_df_virtual_ladder` applies the bare term $\tfrac12\sum_{ef} \tau_{ijef}\langle ab\Vert ef\rangle$ to one auxiliary slice using one metric-solved factor and one raw factor. Same-spin exchange acts on tau before projection. In the mixed sector, the exchange sign cancels the beta-alpha tau permutation sign, yielding the direct contraction. The same TensorIR gives the tau and factor JVPs and dense Frobenius VJPs. The future caller owns physical amplitude packing and factor-to-AO pullback.

## Evidence

FP64 checks against an independently assembled dense spin tensor covered equal, unequal, and empty spin domains and a ragged final auxiliary slice. The maximum residual discrepancy was $1.11\times10^{-16}$. The maximum tau and factor directional discrepancies were each $3.05\times10^{-16}$, with transpose dot discrepancies of $2.22\times10^{-16}$ and $4.44\times10^{-16}$, respectively.

For an eight-auxiliary slice at the exact CH₃Br/ha-T shape, the largest live tensor was 633,025,600 bytes. The static SM89 plans for the isolated primal, tau VJP, and factor VJP used 2,338,500,912, 2,813,270,064, and 3,773,761,328 combined device and host-staging bytes, respectively.

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=python python -B -m pytest -q -p no:cacheprovider tests/python/test_uccsd_df_virtual_ladder.py
python -B tools/check_compiler_structure.py
```

## Revisit gate

At 572 auxiliary functions, the static count for 71 eight-function slices and one four-function slice is $1.2567\times10^{13}$ scalar summands per bare primal and $1.2569\times10^{13}$ per bare tau VJP. Together with the remaining graph, that per-action cost tightly limits Lambda iterations under the archived $2.10\times10^{14}$ total work envelope. Evaluate work as well as simultaneous memory before selecting an execution layout.

Runtime admission requires the full dressed residual, both amplitude actions, staged source pullbacks, and fresh T and Lambda qualification. Compare the full energy and all 15 forces with the retained dense CH₃Br/ha-D endpoint and reuse its same-root energy controls. Then measure whole-endpoint work, memory and time on L20 before qualifying the ha-T endpoint.
