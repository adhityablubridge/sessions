# CP Overlap Race Audit - 2026-06-10

Context: CUDA stream-ordering races in context-parallel ring-attention compute/communication overlap causing grad-norm explosion after ~145 steps. Step-0 gradient parity passes; no-overlap path is stable.

Files audited:
- RingRotator.h
- ContextParallel.h
- ContextParallelBackward.h

See full findings in assistant response for this session.
