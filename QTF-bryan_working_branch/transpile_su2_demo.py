
"""
ansatz_compare.py
=================
Builds TWO ansatze, transpiles both to the same hardware backend,
and prints a side-by-side stats comparison.

Ansatz A — EfficientSU2 (baseline)
    6 qubits | reps=8 | linear entanglement | 108 params

Ansatz B — Custom Brickwork (ours)
    6 qubits | reps=6 | brickwork entanglement | dense encoding
    Layout:
      - Encoding layer : RY(x_i) + RZ(x_i) per qubit  → 12 encode params
      - Per variational rep:
            ZYZ rotation block (RZ-RY-RZ) per qubit    → 18 params/rep
            Brickwork CX layer (even CX pairs)
            ZYZ rotation block                          → 18 params/rep
            Brickwork CX layer (odd CX pairs)
      - reps=3  →  3 × 2 × 18 = 108 variational params
      - Total   : 12 + 108 = 120 params

Credentials via .env:
    IBM_TOKEN=...
    IBM_CRN=crn:v1:...
"""

import os
import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import EfficientSU2
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# ─────────────────────────────────────────────────────────────────────────────
#  BACKEND

QiskitRuntimeService.save_account(
    channel  = "ibm_cloud",
    token    = "wj4YbajDEHX73jZLSuHptbQtgPl1Za0X-Qy95f0Lbbfr",
    instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/813b37ffee14414ca81092ab94341434:1284900f-4e18-41c7-aadf-44278c5d44da::",
    overwrite= True,
)

service = QiskitRuntimeService(
    channel  = "ibm_cloud",
    token    = "wj4YbajDEHX73jZLSuHptbQtgPl1Za0X-Qy95f0Lbbfr",
    instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/813b37ffee14414ca81092ab94341434:1284900f-4e18-41c7-aadf-44278c5d44da::",
)
backend = service.least_busy(min_num_qubits=6, operational=True)

print("=" * 65)
print("  BACKEND")
print("=" * 65)
print(f"  Name       : {backend.name}")
try:
    cfg = backend.configuration()
    print(f"  Qubits     : {cfg.n_qubits}")
    print(f"  Basis gates: {cfg.basis_gates}")
    print(f"  Simulator  : {cfg.simulator}")
except Exception:
    print(f"  Qubits     : {backend.num_qubits}")
    print(f"  Basis gates: {list(backend.operation_names)}")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def circuit_stats(qc_t: QuantumCircuit, label: str) -> dict:
    """Compute and print circuit stats for a transpiled circuit."""
    ops   = dict(qc_t.count_ops())
    cx    = ops.get("cx",  0)
    ecr   = ops.get("ecr", 0)
    cz    = ops.get("cz",  0)
    two_q = cx + ecr + cz

    # 2-qubit-only depth
    qc_2q = QuantumCircuit(qc_t.num_qubits)
    for inst in qc_t.data:
        if len(inst.qubits) == 2:
            qc_2q.append(inst.operation, inst.qubits)
    two_q_depth = qc_2q.depth()

    excluded = {"cx", "ecr", "cz", "measure", "barrier"}
    one_q = sum(v for k, v in ops.items() if k not in excluded)

    stats = {
        "label"         : label,
        "total_depth"   : qc_t.depth(),
        "two_q_depth"   : two_q_depth,
        "total_gates"   : sum(ops.values()),
        "two_q_gates"   : two_q,
        "cx"            : cx,
        "ecr"           : ecr,
        "one_q_gates"   : one_q,
        "t_gates"       : ops.get("t", 0),
        "num_qubits"    : qc_t.num_qubits,
        "num_params"    : qc_t.num_parameters,
        "ops"           : ops,
    }
    return stats


def print_stats(s: dict):
    print(f"\n  ── {s['label']} ──")
    print(f"  Num qubits         : {s['num_qubits']}")
    print(f"  Num parameters     : {s['num_params']}")
    print(f"  Total depth        : {s['total_depth']}")
    print(f"  2-qubit gate depth : {s['two_q_depth']}")
    print(f"  Total gates        : {s['total_gates']}")
    print(f"  2-qubit gates      : {s['two_q_gates']}  (CX={s['cx']}  ECR={s['ecr']})")
    print(f"  1-qubit gates      : {s['one_q_gates']}")
    print(f"  T-gate count       : {s['t_gates']}")
    print(f"  Gate breakdown     : {s['ops']}")


def print_comparison(sA: dict, sB: dict):
    """Side-by-side delta comparison."""
    metrics = [
        ("Total depth",        "total_depth"),
        ("2-qubit gate depth", "two_q_depth"),
        ("Total gates",        "total_gates"),
        ("2-qubit gates",      "two_q_gates"),
        ("1-qubit gates",      "one_q_gates"),
        ("T-gate count",       "t_gates"),
    ]
    w = 28
    print("\n" + "═"*72)
    print("  SIDE-BY-SIDE COMPARISON")
    print("═"*72)
    print(f"  {'Metric':<{w}} {'EfficientSU2':>12}  {'Brickwork':>12}  {'Delta':>10}")
    print("  " + "-"*68)
    for label, key in metrics:
        a, b  = sA[key], sB[key]
        delta = b - a
        sign  = "+" if delta >= 0 else ""
        better = "▲ worse" if delta > 0 else ("▼ better" if delta < 0 else "  same")
        print(f"  {label:<{w}} {a:>12}  {b:>12}  {sign}{delta:>6}  {better}")
    print("═"*72)


# ─────────────────────────────────────────────────────────────────────────────
#  ANSATZ A — EfficientSU2
# ─────────────────────────────────────────────────────────────────────────────
def build_efficient_su2(n_qubits=6, reps=8) -> QuantumCircuit:
    """Standard EfficientSU2 baseline — 108 params."""
    ansatz = EfficientSU2(
        num_qubits   = n_qubits,
        entanglement = "linear",
        reps         = reps,
    )
    assert ansatz.num_parameters == 108
    return ansatz


# ─────────────────────────────────────────────────────────────────────────────
#  ANSATZ B — Custom Brickwork
# ─────────────────────────────────────────────────────────────────────────────
def build_brickwork_ansatz(n_qubits=6, var_reps=3) -> QuantumCircuit:
    """
    Custom brickwork ansatz.

    Structure
    ---------
    1. Dense encoding layer:
         For each qubit i: RY(enc[2i]) → RZ(enc[2i+1])
         → 2 × n_qubits = 12 encoding parameters

    2. Variational block (repeated var_reps times):
         a. ZYZ layer  : RZ-RY-RZ per qubit  (3 × n_qubits params)
         b. Even brick : CX on (0,1), (2,3), (4,5)
         c. ZYZ layer  : RZ-RY-RZ per qubit  (3 × n_qubits params)
         d. Odd brick  : CX on (1,2), (3,4)
       → 2 × 3 × 6 = 36 params per rep × 3 reps = 108 variational

    Total: 12 + 108 = 120 parameters
    """
    n_enc = 2 * n_qubits                        # 12
    n_var = var_reps * 2 * 3 * n_qubits         # 108

    enc = ParameterVector("enc", n_enc)
    var = ParameterVector("var", n_var)

    qc = QuantumCircuit(n_qubits)
    p  = 0  # var parameter index

    # ── 1. Dense encoding ────────────────────────────────────────────────────
    qc.barrier(label="encode")
    for i in range(n_qubits):
        qc.ry(enc[2*i],   i)
        qc.rz(enc[2*i+1], i)

    # ── 2. Variational reps ───────────────────────────────────────────────────
    for rep in range(var_reps):
        qc.barrier(label=f"rep{rep}a")

        # ZYZ block before even CX
        for i in range(n_qubits):
            qc.rz(var[p],   i); p += 1
            qc.ry(var[p],   i); p += 1
            qc.rz(var[p],   i); p += 1

        # Even brickwork CX: (0,1), (2,3), (4,5)
        qc.barrier(label=f"rep{rep}_even")
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)

        # ZYZ block before odd CX
        qc.barrier(label=f"rep{rep}b")
        for i in range(n_qubits):
            qc.rz(var[p],   i); p += 1
            qc.ry(var[p],   i); p += 1
            qc.rz(var[p],   i); p += 1

        # Odd brickwork CX: (1,2), (3,4)
        qc.barrier(label=f"rep{rep}_odd")
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)

    assert p == n_var, f"Var param mismatch: expected {n_var}, got {p}"
    assert qc.num_parameters == n_enc + n_var, \
        f"Total param mismatch: expected {n_enc+n_var}, got {qc.num_parameters}"

    return qc


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD, TRANSPILE, STATS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Building circuits …")
circ_A = build_efficient_su2(n_qubits=6, reps=8)
circ_B = build_brickwork_ansatz(n_qubits=6, var_reps=3)

print(f"[INFO] EfficientSU2  params : {circ_A.num_parameters}")
print(f"[INFO] Brickwork     params : {circ_B.num_parameters}")

print("[INFO] Transpiling A (EfficientSU2) …")
qc_tA = transpile(circ_A, backend=backend, optimization_level=3, seed_transpiler=42)

print("[INFO] Transpiling B (Brickwork) …")
qc_tB = transpile(circ_B, backend=backend, optimization_level=3, seed_transpiler=42)

statsA = circuit_stats(qc_tA, "EfficientSU2  (A)")
statsB = circuit_stats(qc_tB, "Brickwork     (B)")

print("\n" + "═"*65)
print("  INDIVIDUAL STATS")
print("═"*65)
print_stats(statsA)
print_stats(statsB)

print_comparison(statsA, statsB)

# ─────────────────────────────────────────────────────────────────────────────
#  EXECUTE BOTH
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

def bind_and_measure(qc_t, n_params):
    vals    = rng.uniform(-np.pi, np.pi, n_params)
    qc_b    = qc_t.assign_parameters(dict(zip(qc_t.parameters, vals)))
    qc_meas = qc_b.copy()
    qc_meas.measure_all()
    return qc_meas

qc_mA = bind_and_measure(qc_tA, circ_A.num_parameters)
qc_mB = bind_and_measure(qc_tB, circ_B.num_parameters)

print(f"\n[INFO] Submitting both circuits to {backend.name} (shots=1024 each) …")
sampler = Sampler(backend=backend)
job_A   = sampler.run([qc_mA], shots=1024)
job_B   = sampler.run([qc_mB], shots=1024)

print(f"[INFO] Job A ID : {job_A.job_id()}")
print(f"[INFO] Job B ID : {job_B.job_id()}")

print("[INFO] Waiting for results …")
res_A = job_A.result()
res_B = job_B.result()

def print_results(result, label):
    counts = result[0].data.meas.get_counts()
    top10  = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  ── {label} ──")
    print(f"  Unique bitstrings : {len(counts)}")
    for bitstr, cnt in top10:
        bar = "█" * int(cnt / 1024 * 40)
        print(f"    |{bitstr}⟩  {cnt:4d}  {bar}")

print("\n" + "═"*65)
print("  EXECUTION RESULTS  (shots=1024)")
print("═"*65)
print_results(res_A, "EfficientSU2  (A)")
print_results(res_B, "Brickwork     (B)")
print("═"*65)

print("\n[DONE]")


