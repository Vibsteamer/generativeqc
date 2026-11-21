# Retain the fixed-amplitude remainder during RHF Λ replay

Implemented an explicit response coordinator, 2026-11-21.

At one fixed-reference and fixed-amplitude epoch, every nonseed input of the remainder transpose is fixed. The existing dependency proof identifies the live invariant values that cross into the seed-dependent graph. Materialize that frontier once under an explicit retained-byte budget. Bind its owned outputs into subsequent remainder actions. Add the prepared spatial virtual-pair transpose and return one physical packed transpose action. The coordinator snapshots the packed amplitudes and requires borrowed equation feeds and spatial operators to be immutable during the epoch.

This supersedes the projected invariant replay in [the spatial-pair study](2026-11-21-uccsd-rhf-spatial-pair-reuse.md). The generated-adjoint program includes inactive definitions for provenance. Only live arithmetic enters preparation and replay, as in ordinary TensorIR execution. Programs with explicit precision execution are excluded.

## Evidence

Synthetic FP64 RHF comparisons apply three physical dual seeds to one prepared amplitude epoch and compare each complete transpose to the dense F12b reference. The test also overwrites the caller's working amplitude vector after preparation and checks admission at an insufficient frontier limit.

The exact CH₃Br/ha-T graph retains 1,064,251,672 bytes. Its static SM89 preparation allocates 10,674,424,832 device and 7,464,868,816 staging bytes. The remainder replay plans 14,452,749,824 device and 7,965,598,720 staging bytes. It counts 4,038,384,805,616 scalar operations. The spatial bare transpose adds 497,690,320,750. Using the conservative full stationarity action and the 1,084,789,006,470 one-time frontier operations, the original $2.10\times10^{14}$ envelope covers 44 transpose actions in this model. The projected count excludes later source and nuclear costs, and it neither predicts iterations nor measures physical GPU time.

## Revisit gate

Use the coordinator as an explicit candidate action inside the existing stationary owner. It still needs reference identity binding, complete simultaneous memory and work admission, independent ha-D energy and Λ agreement, and a molecular replay of all 15 force components. The public path keeps its existing authoritative qualification until those gates pass.
