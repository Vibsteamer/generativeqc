# Context ownership of RCCSD Lambda diagnostics

Status: implemented
Date: 2026-11-26

The focal force endpoints publish a snapshot of the calling thread's completed Lambda diagnostic into their context before releasing its mutex. Each snapshot owns the scalar fields, counts and handoff text exposed by the v1 C getter. Reset discards that snapshot. Destroying the context releases it even for native callers that never reset diagnostics.

A process-wide registry keyed by the context address could return a previous handle's residual after allocator reuse. Its FIFO limit could also evict results from live contexts, and resetting a key left its queue entry behind. Context ownership bounds retained diagnostic storage by the number of handles. Thread-local storage also supports CPU solves whose C entry point takes no context.

A context-bound Python caller treats missing context telemetry as unavailable. A thread-only library cannot establish which context produced a residual. The native audit exercises lifetime reuse, more than 1,024 simultaneous contexts and interleaved reads from different threads. The Python endpoint regression covers an older library offering a valid thread diagnostic.
