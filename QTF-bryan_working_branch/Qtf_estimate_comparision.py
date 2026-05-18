#!/usr/bin/env python3
"""
QTF Full Transpilation Benchmark
==================================
Runs actual transpilation on a real IBM backend (with coupling map)
for all 11 QTF sequences and reports exact circuit resources.

USAGE:
    python qtf_transpile_all.py

CREDENTIALS:
    Already saved via QiskitRuntimeService.save_account()
    or set: export QTF_IBM_TOKEN="..." and export QTF_IBM_INSTANCE="..."
"""

import os
import sys
import time
import json
import math
import numpy as np

# ── Qiskit imports ──────────────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import EfficientSU2
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.circuit import ParameterVector

except ImportError as e:
    print(f"[ERROR] {e}")
    sys.exit(1)

# ── QTF runner ──────────────────────────────────────────────────────────────────
try:
    import QTF.runner_simulator as runner
except ImportError:
    print("[ERROR] Could not import QTF.runner_simulator. Run from project root.")
    sys.exit(1)

QiskitRuntimeService.save_account(
        channel="ibm_cloud",
        token="wj4YbajDEHX73jZLSuHptbQtgPl1Za0X-Qy95f0Lbbfr",
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/813b37ffee14414ca81092ab94341434:cc2430d9-437f-4ac7-bf1d-656af1cfca82::",
        set_as_default=True,
        overwrite=True,
    )

# ==============================================================================
# CONFIG
# ==============================================================================

SEQUENCES = {
    "5AWL": "YYDPETGTWY",
    "2MZX": "DLDALLADLE",
    "1K43": "RGKWTYNGITYEGR",
    "8T61": "RHYYKFNSTGRHYHYY",
    "8T63": "WHMWNTVPNAKQVIAA",
    "2NDC": "GGLRSLGRKILRAWKKYG",
    "2NDE": "IGLRGLGRKIALIHKKYG",
    "2JOF": "DAYAQWLKDGGPSSGRPPPS",
    "6A8Y": "YYHFWHRGVTKRSLSPHRPRHSRLQR",
    "1G04": "GNDYEDRYYRENMYRYPNQVYYRPVC",
    "2KNC": "GAMGSEERAIPIWWVLVGVLGGLLLLTILVLAMWKVGFFKRNRPPLEEDDEEGE",
}

SELECTIVE_CHI_MAP = {
    "Y": ["chi1", "chi2"], "W": ["chi1", "chi2"],
    "F": ["chi1", "chi2"], "H": ["chi1", "chi2"],
    "D": ["chi1"], "E": ["chi1"], "N": ["chi1"], "Q": ["chi1"],
    "T": ["chi1"], "S": ["chi1"], "V": ["chi1"], "I": ["chi1"],
    "L": ["chi1"], "M": ["chi1"], "K": ["chi1"], "R": ["chi1"],
    "C": ["chi1"], "P": ["chi1"], "A": [],        "G": [],
}

OPTIMIZATION_LEVEL = 3
TWO_QUBIT_GATES    = {"cx", "cz", "ecr", "rzz", "rxx", "ryy", "cp", "swap", "iswap"}

SHOTS_PER_CIRCUIT  = 4096   # for hardware timing estimate
JOB_OVERHEAD_S     = 2.0   # IBM submission overhead per job

# ==============================================================================
# DROP THIS BELOW YOUR EXISTING CONFIG BLOCK
# Requires: runner, QiskitRuntimeService, transpile, QuantumCircuit,
#           ParameterVector, EfficientSU2 — all already imported in your file
# ==============================================================================

# ── Ansatz A: EfficientSU2 ────────────────────────────────────────────────────
def build_efficient_su2(n_qubits: int, reps: int) -> QuantumCircuit:
    """Standard EfficientSU2 — linear entanglement, RY+RZ rotations."""
    return EfficientSU2(
        num_qubits   = n_qubits,
        entanglement = "linear",
        reps         = reps,
    )


# ── Ansatz B: Custom Brickwork ────────────────────────────────────────────────
def build_brickwork_ansatz(n_qubits: int, reps: int) -> QuantumCircuit:
    """
    Custom brickwork ansatz — alternating parallel CX layers.

    Per rep:
      [RY+RZ all qubits]
      Even CX: (0,1),(2,3),(4,5),...   ← all parallel, depth=1
      [RY+RZ all qubits]
      Odd  CX: (1,2),(3,4),(5,6),...   ← all parallel, depth=1
    Final [RY+RZ all qubits]

    Same param count as EfficientSU2 with same n_qubits/reps:
        2 * n_qubits * (reps + 1)
    """
    n_params = 2 * n_qubits * (reps + 1)
    params   = ParameterVector("θ", n_params)
    qc       = QuantumCircuit(n_qubits)
    p        = 0

    for _ in range(reps):
        for q in range(n_qubits):
            qc.ry(params[p], q); p += 1
            qc.rz(params[p], q); p += 1
        for q in range(0, n_qubits - 1, 2):   # even pairs
            qc.cx(q, q + 1)
        for q in range(n_qubits):
            qc.ry(params[p], q); p += 1
            qc.rz(params[p], q); p += 1
        for q in range(1, n_qubits - 1, 2):   # odd pairs
            qc.cx(q, q + 1)

    # final rotation layer
    for q in range(n_qubits):
        qc.ry(params[p], q); p += 1
        qc.rz(params[p], q); p += 1

    return qc


# ── Build folder (uses your existing runner + SELECTIVE_CHI_MAP) ──────────────
def build_folder(sequence: str) -> runner.QuantumBiophysicsFolder:
    return runner.QuantumBiophysicsFolder(
        sequence,
        force_field      = "amber",
        chi_mode         = "selective",
        selective_chi_map= SELECTIVE_CHI_MAP,
    )


# ── Transpile one circuit, return all metrics ─────────────────────────────────
def _transpile_circuit(qc: QuantumCircuit, backend, label: str) -> dict:
    t0  = time.time()
    tqc = transpile(qc, backend=backend,
                    optimization_level=OPTIMIZATION_LEVEL,
                    seed_transpiler=42)
    elapsed = time.time() - t0

    op_counts = dict(tqc.count_ops())
    n_2q      = sum(v for k, v in op_counts.items()
                    if k.lower() in TWO_QUBIT_GATES)
    depth     = tqc.depth()

    # 2-qubit-only depth
    tqc_2q = tqc.copy_empty_like()
    for inst in tqc.data:
        if inst.operation.name.lower() in TWO_QUBIT_GATES:
            tqc_2q.append(inst)
    two_q_depth = tqc_2q.depth()

    # Hardware timing estimate (final shot — 3 basis circuits Z/X/Y)
    try:
        dt = backend.dt if (hasattr(backend, "dt") and backend.dt) else 0.5e-9
    except Exception:
        dt = 0.5e-9
    circuit_exec_s    = (depth * dt * 1e6 + 5.0) * SHOTS_PER_CIRCUIT * 1e-6
    hw_time_per_rep_s = 3 * (circuit_exec_s + JOB_OVERHEAD_S)

    noise = ("low"      if two_q_depth < 100 else
             "marginal" if two_q_depth < 200 else "high")

    return {
        "label":                   label,
        "n_params":                qc.num_parameters,
        "depth_original":          qc.depth(),
        "depth_transpiled":        depth,
        "two_qubit_depth":         two_q_depth,
        "two_qubit_count":         n_2q,
        "gates_transpiled":        sum(op_counts.values()),
        "op_breakdown":            op_counts,
        "transpile_time_s":        round(elapsed, 3),
        "hw_time_per_replica_s":   round(hw_time_per_rep_s, 2),
        "hw_time_per_replica_min": round(hw_time_per_rep_s / 60, 3),
        "noise":                   noise,
    }


# ── Benchmark one sequence with both ansatze ──────────────────────────────────
def benchmark_sequence(pdb: str, seq: str, backend) -> dict:
    folder = build_folder(seq)

    # Build both circuits with the same n_qubits / reps from QTF folder
    qc_su2   = build_efficient_su2(folder.n_qubits, folder.reps)
    qc_brick = build_brickwork_ansatz(folder.n_qubits, folder.reps)

    m_su2   = _transpile_circuit(qc_su2,   backend, "EfficientSU2")
    m_brick = _transpile_circuit(qc_brick, backend, "Brickwork")

    return {
        "pdb":          pdb,
        "sequence":     seq,
        "len":          len(seq),
        "n_qubits":     folder.n_qubits,
        "reps":         folder.reps,
        "total_angles": folder.total_angles,
        "n_params":     folder.n_params,
        "backend":      backend.name,
        "su2":          m_su2,
        "brickwork":    m_brick,
        "delta": {
            "depth":       m_brick["depth_transpiled"] - m_su2["depth_transpiled"],
            "two_q_depth": m_brick["two_qubit_depth"]  - m_su2["two_qubit_depth"],
            "two_q_count": m_brick["two_qubit_count"]  - m_su2["two_qubit_count"],
        },
    }


# ── Print comparison table ────────────────────────────────────────────────────
def print_summary(results: list):
    sep = "=" * 130
    sign = lambda x: f"+{x}" if x > 0 else str(x)

    print(f"\n{sep}")
    print(f"  QTF ANSATZ COMPARISON  |  backend: {results[0]['backend']}"
          f"  |  opt_level={OPTIMIZATION_LEVEL}")
    print(sep)
    print(f"{'PDB':<8} {'Len':>4} {'Q':>3} {'R':>4} {'Params':>7}  "
          f"{'── EfficientSU2 ──':^30}  {'──── Brickwork ────':^30}  "
          f"{'── Delta (BW−SU2) ──':^28}  {'Verdict'}")
    print(f"{'':8} {'':4} {'':3} {'':4} {'':7}  "
          f"{'depth':>8} {'2Qdepth':>9} {'2Qcount':>9}  "
          f"{'depth':>8} {'2Qdepth':>9} {'2Qcount':>9}  "
          f"{'Δdepth':>7} {'Δ2Qdepth':>9} {'Δ2Qcnt':>8}  {'':}")
    print("-" * 130)

    for r in results:
        s = r["su2"]; b = r["brickwork"]; d = r["delta"]
        verdict = ("BW ▼ better" if d["two_q_depth"] < 0
                   else "SU2 ▼ better" if d["two_q_depth"] > 0
                   else "tied")
        print(
            f"{r['pdb']:<8} {r['len']:>4} {r['n_qubits']:>3} {r['reps']:>4} "
            f"{r['n_params']:>7}  "
            f"{s['depth_transpiled']:>8} {s['two_qubit_depth']:>9} {s['two_qubit_count']:>9}  "
            f"{b['depth_transpiled']:>8} {b['two_qubit_depth']:>9} {b['two_qubit_count']:>9}  "
            f"{sign(d['depth']):>7} {sign(d['two_q_depth']):>9} {sign(d['two_q_count']):>8}  "
            f"{verdict}"
        )

    print(sep)

    print("\n── Gate breakdown ────────────────────────────────────────────────────")
    for r in results:
        print(f"\n  {r['pdb']}  ({r['len']} aa | {r['n_qubits']}q | reps={r['reps']})")
        print(f"    SU2      : {r['su2']['op_breakdown']}")
        print(f"    Brickwork: {r['brickwork']['op_breakdown']}")
        print(f"    Δ 2Q depth={r['delta']['two_q_depth']:+d}  "
              f"Δ 2Q count={r['delta']['two_q_count']:+d}  "
              f"Δ depth={r['delta']['depth']:+d}")
    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    service = QiskitRuntimeService()

    print("\n── Available Backends ────────────────────────────────────────────────")
    all_backends = service.backends(simulator=False, operational=True)
    backends_sorted = sorted(all_backends, key=lambda b: b.status().pending_jobs)
    for b in backends_sorted:
        st = b.status()
        print(f"  {b.name:<30} {b.num_qubits:>6} qubits  "
              f"{st.pending_jobs:>6} pending  {st.status_msg}")
    print("─" * 70)

    backend = backends_sorted[0]
    print(f"\n[INFO] Using: {backend.name} "
          f"({backend.num_qubits} qubits, {backend.status().pending_jobs} pending)\n")

    results = []
    for idx, (pdb, seq) in enumerate(SEQUENCES.items(), 1):
        print(f"[{idx:>2}/{len(SEQUENCES)}] {pdb:<8} len={len(seq):>3}", end="  ", flush=True)
        try:
            r = benchmark_sequence(pdb, seq, backend)
            results.append(r)
            print(f"SU2 2Qdepth={r['su2']['two_qubit_depth']:>4}  "
                  f"BW 2Qdepth={r['brickwork']['two_qubit_depth']:>4}  "
                  f"Δ={r['delta']['two_q_depth']:+d}  "
                  f"noise_su2={r['su2']['noise']}  noise_bw={r['brickwork']['noise']}")
        except Exception as e:
            print(f"ERROR: {e}")

    clean = [r for r in results if "error" not in r]
    if clean:
        print_summary(clean)

    # Save
    with open("qtf_transpile_benchmark.json", "w") as f:
        json.dump(results, f, indent=4)

    cols = ["pdb","len","n_qubits","reps","n_params","total_angles","backend",
            "su2_depth","su2_2q_depth","su2_2q_count","su2_noise",
            "bw_depth","bw_2q_depth","bw_2q_count","bw_noise",
            "delta_depth","delta_2q_depth","delta_2q_count"]
    with open("qtf_transpile_benchmark.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in clean:
            f.write(",".join([
                r["pdb"], str(r["len"]), str(r["n_qubits"]), str(r["reps"]),
                str(r["n_params"]), str(r["total_angles"]), r["backend"],
                str(r["su2"]["depth_transpiled"]),
                str(r["su2"]["two_qubit_depth"]),
                str(r["su2"]["two_qubit_count"]),
                r["su2"]["noise"],
                str(r["brickwork"]["depth_transpiled"]),
                str(r["brickwork"]["two_qubit_depth"]),
                str(r["brickwork"]["two_qubit_count"]),
                r["brickwork"]["noise"],
                str(r["delta"]["depth"]),
                str(r["delta"]["two_q_depth"]),
                str(r["delta"]["two_q_count"]),
            ]) + "\n")

    print("[INFO] Saved → qtf_transpile_benchmark.json")
    print("[INFO] Saved → qtf_transpile_benchmark.csv")


if __name__ == "__main__":
    main()