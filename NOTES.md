# QTF Notes

## PHEAT Backbone Geometry Angles and Qubit Requirements

Adding PHEAT `omega`, `tau`, and `theta` values as stored-only output or report
fields does not change QTF qubit requirements.

If those values become QTF-generated or optimized degrees of freedom, they add
`3N - 2` angles for a sequence of length `N`:

```text
T_new = T_current + 3N - 2
n_qubits = max(2, ceil(log2(total_angles)))
```

For `YYDPETGTWY` under the current `runner_hardware3.py` selective-chi behavior,
the current count is `33` angles and `6` qubits. Adding all three geometry angle
types gives `61` angles, which still fits in `6` qubits because `2^6 = 64`.
However, the runner rep heuristic increases reps from `3` to `6`, raising
trainable parameters from `48` to `84`.
