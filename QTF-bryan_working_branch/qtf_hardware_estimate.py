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
# HELPERS
# ==============================================================================

def build_folder(sequence: str) -> runner.QuantumBiophysicsFolder:
    return runner.QuantumBiophysicsFolder(
        sequence,
        force_field="amber",
        chi_mode="selective",
        selective_chi_map=SELECTIVE_CHI_MAP,
    )


def transpile_for_backend(folder, backend) -> dict:
    """
    Constructs QuantumCircuit, appends ansatz, transpiles to real backend
    with its full coupling map and native gate set.
    Returns a dict of all resource metrics.
    """
    ansatz = folder.ansatz

    # ── Build circuit ──────────────────────────────────────────────────────────
    qc = QuantumCircuit(ansatz.num_qubits)
    qc.append(ansatz, range(ansatz.num_qubits))
    qc = qc.decompose()  # flatten into primitive gates

    # ── Transpile with real backend coupling map ───────────────────────────────
    t0  = time.time()
    tqc = transpile(qc, backend=backend, optimization_level=OPTIMIZATION_LEVEL)
    transpile_time = time.time() - t0

    # ── Gate counts ────────────────────────────────────────────────────────────
    op_counts  = tqc.count_ops()
    n_2q_total = sum(v for k, v in op_counts.items() if k.lower() in TWO_QUBIT_GATES)

    # ── 2-qubit gate depth ─────────────────────────────────────────────────────
    # Filter to only 2Q gates, measure depth of that subcircuit
    tqc_2q = tqc.copy_empty_like()
    for inst in tqc.data:
        if inst.operation.name.lower() in TWO_QUBIT_GATES:
            tqc_2q.append(inst)
    two_qubit_depth = tqc_2q.depth()

    # ── Hardware time estimate (final shot only per replica) ───────────────────
    try:
        dt = backend.dt if (hasattr(backend, "dt") and backend.dt) else 0.5e-9
    except Exception:
        dt = 0.5e-9
    depth              = tqc.depth()
    readout_us         = 5.0
    circuit_exec_s     = (depth * dt * 1e6 + readout_us) * SHOTS_PER_CIRCUIT * 1e-6
    hw_time_per_rep_s  = 3 * (circuit_exec_s + JOB_OVERHEAD_S)  # 3 bases Z/X/Y

    # ── Noise verdict ──────────────────────────────────────────────────────────
    if two_qubit_depth < 100:
        noise = "low"
    elif two_qubit_depth < 200:
        noise = "marginal"
    else:
        noise = "high"

    return {
        # Circuit identity
        "n_qubits":           folder.n_qubits,
        "reps":               folder.reps,
        "n_params":           folder.n_params,
        "total_angles":       folder.total_angles,

        # Pre-transpilation
        "depth_original":     ansatz.depth(),
        "gates_original":     sum(ansatz.count_ops().values()),

        # Post-transpilation (real hardware coupling map)
        "depth_transpiled":   depth,
        "two_qubit_depth":    two_qubit_depth,
        "gates_transpiled":   sum(op_counts.values()),
        "two_qubit_count":    n_2q_total,
        "op_breakdown":       dict(op_counts),
        "transpile_time_s":   round(transpile_time, 3),

        # Hardware timing estimate
        "hw_time_per_replica_s":  round(hw_time_per_rep_s, 2),
        "hw_time_per_replica_min": round(hw_time_per_rep_s / 60, 3),

        # Verdict
        "noise": noise,
    }


def print_summary(results: list):
    sep = "=" * 120

    print(f"\n{sep}")
    print(f"  QTF FULL TRANSPILATION BENCHMARK  |  backend: {results[0]['backend']}  |  opt_level={OPTIMIZATION_LEVEL}")
    print(sep)

    header = (
        f"{'PDB':<8} {'Sequence':<55} {'Len':>4} {'Angles':>7} "
        f"{'Qubits':>7} {'Reps':>5} {'Params':>7} "
        f"{'Depth':>7} {'2Q depth':>9} {'2Q count':>9} "
        f"{'HW time/rep':>12}  {'Noise':<10}  {'Transpile(s)':>13}"
    )
    print(header)
    print("-" * 120)

    for r in results:
        seq_disp = r["sequence"][:54]
        print(
            f"{r['pdb']:<8} {seq_disp:<55} {r['len']:>4} {r['total_angles']:>7} "
            f"{r['n_qubits']:>7} {r['reps']:>5} {r['n_params']:>7} "
            f"{r['depth_transpiled']:>7} {r['two_qubit_depth']:>9} {r['two_qubit_count']:>9} "
            f"{r['hw_time_per_replica_s']:>10.1f}s  {r['noise']:<10}  {r['transpile_time_s']:>13.3f}"
        )

    print(sep)

    # Per-sequence gate breakdown
    print("\n── Gate breakdown per sequence ───────────────────────────────────────────────────")
    for r in results:
        print(f"  {r['pdb']:<8}  {r['op_breakdown']}")

    print(sep + "\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # ── Connect ─────────────────────────────────────────────────────────────────
    print("[INFO] Connecting to IBM Quantum...")
    service = QiskitRuntimeService()

    # ── List all backends ───────────────────────────────────────────────────────
    print("\n── Available Backends ────────────────────────────────────────────────")
    all_backends = service.backends(simulator=False, operational=True)
    backends_sorted = sorted(all_backends, key=lambda b: b.status().pending_jobs)
    for b in backends_sorted:
        st = b.status()
        print(f"  {b.name:<30} {b.num_qubits:>6} qubits   {st.pending_jobs:>6} pending   {st.status_msg}")
    print("─" * 70)

    # ── Pick least busy ─────────────────────────────────────────────────────────
    #backend = backends_sorted[0]
    backend_name = "ibm_miami"  # Change to your preferred backend
    backend = service.backend(backend_name)
    print(f"\n[INFO] Using backend: {backend.name} "
          f"({backend.num_qubits} qubits, {backend.status().pending_jobs} pending)\n")

    # ── Run transpilation for each sequence ────────────────────────────────────
    results = []
    total_sequences = len(SEQUENCES)

    for idx, (pdb, seq) in enumerate(SEQUENCES.items(), 1):
        print(f"[{idx:>2}/{total_sequences}] {pdb}  ({seq[:30]}{'...' if len(seq)>30 else ''})  "
              f"len={len(seq)}", end="  ", flush=True)

        try:
            folder  = build_folder(seq)
            metrics = transpile_for_backend(folder, backend)

            result = {
                "pdb":      pdb,
                "sequence": seq,
                "len":      len(seq),
                "backend":  backend.name,
                **metrics,
            }
            results.append(result)
            print(f"depth={metrics['depth_transpiled']}  "
                  f"2Q_depth={metrics['two_qubit_depth']}  "
                  f"noise={metrics['noise']}  "
                  f"({metrics['transpile_time_s']}s)")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "pdb": pdb, "sequence": seq, "len": len(seq),
                "backend": backend.name, "error": str(e),
            })

    # ── Print summary table ─────────────────────────────────────────────────────
    clean_results = [r for r in results if "error" not in r]
    if clean_results:
        print_summary(clean_results)

    # ── Save JSON ───────────────────────────────────────────────────────────────
    out_path = "qtf_transpile_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Full results saved to: {out_path}")

    # ── CSV ─────────────────────────────────────────────────────────────────────
    csv_path = "qtf_transpile_benchmark.csv"
    with open(csv_path, "w") as f:
        cols = [
            "pdb","sequence","len","n_qubits","reps","n_params","total_angles",
            "depth_original","depth_transpiled","two_qubit_depth","two_qubit_count",
            "gates_transpiled","hw_time_per_replica_s","noise","transpile_time_s","backend"
        ]
        f.write(",".join(cols) + "\n")
        for r in results:
            if "error" in r:
                continue
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[INFO] CSV saved to: {csv_path}\n")


if __name__ == "__main__":
    main()