# Molecular F12 library ownership

Status: implemented
Date: 2026-11-26

The shell integral adjoints and the relaxed molecular HF+CABS response have
different backend contracts. The shell source set supplies explicit CPU
reference entry points as well as device kernels. The response contracts call
cuSOLVER and cuBLAS on device. Keep independent targets so a CUDA provider
cannot silently bind a host shell oracle.

CMake builds both targets from owned sources in `src/posthf` and the shared
solver adapters. The shell target compiles the Coulomb and geminal operators
in separate translation units, with generated headers specialized to the
selected maximum angular momentum and backend. The same build graph supplies
SDK and wheel products. Public ELF export lists under `manifests/f12` keep
internal C++ and CUDA implementation symbols private. NVIDIA wheel builds
use the same lazy provider imports as the main runtime, preserving the
ability to load the package before a GPU is selected.

The source-checkout shell builder supports the smaller qualification domains
through the same CMake targets. The installed molecular provider uses the
through-i domain. See the molecular F12 recipe in `docs/user/ccsdt_cbs.md`
for SDK build and installation commands.
