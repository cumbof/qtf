#!/usr/bin/env python3
"""
QTF Hardware Resource Estimator
================================
Estimates IBM Quantum hardware resources BEFORE running the full folding pipeline.
Outputs:
  - Circuit depth after transpilation
  - Two-qubit gate count after transpilation
  - Estimated wall-clock time for the full program on real hardware
  - Least-busy backend selection

USAGE:
    python qtf_hardware_estimate.py --predict "YYDPETGTWY" --forcefield amber

CREDENTIALS:
    Set environment variables (do NOT hardcode tokens):
        export QTF_IBM_TOKEN="your_token_here"
        export QTF_IBM_INSTANCE="crn:v1:bluemix:..."

    Or pass via CLI:
        python qtf_hardware_estimate.py --predict "YYDPETGTWY" --ibm_token "..." --ibm_instance "..."
"""

import os
import sys
import time
import argparse
import numpy as np

# ── Qiskit imports ─────────────────────────────────────────────────────────────
try:
    from qiskit.circuit.library import EfficientSU2
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService
    QISKIT_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Qiskit import failed: {e}")
    print("Install with: pip install qiskit qiskit-ibm-runtime qiskit-aer")
    sys.exit(1)

# ── QTF runner import ──────────────────────────────────────────────────────────
try:
    import QTF.runner_simulator as runner
except ImportError:
    print("[ERROR] Could not import QTF.runner_simulator. Make sure you run from the project root.")
    sys.exit(1)


# ==============================================================================
# 1. BACKEND SELECTOR  –  pick the least busy real device
# ==============================================================================

def get_least_busy_backend(service: QiskitRuntimeService, min_qubits: int = 5):
    """
    Returns the least-busy operational backend with at least `min_qubits` qubits.
    Filters out simulators and offline systems.
    """
    print(f"\n[INFO] Querying IBM Quantum backends (min_qubits={min_qubits})...")

    backends = service.backends(
        simulator=False,
        operational=True,
        min_num_qubits=min_qubits,
    )

    if not backends:
        raise RuntimeError(
            f"No operational real backends found with >= {min_qubits} qubits."
        )

    # Sort by pending jobs (least busy first)
    backends_sorted = sorted(backends, key=lambda b: b.status().pending_jobs)

    print(f"\n{'Backend':<25} {'Qubits':>6} {'Pending Jobs':>13} {'Status'}")
    print("-" * 60)
    for b in backends_sorted[:8]:   # show top 8
        st = b.status()
        marker = " <-- SELECTED" if b == backends_sorted[0] else ""
        print(f"  {b.name:<23} {b.num_qubits:>6} {st.pending_jobs:>13}    {st.status_msg}{marker}")

    selected = backends_sorted[0]
    print(f"\n[INFO] Selected backend: {selected.name} "
          f"({selected.num_qubits} qubits, "
          f"{selected.status().pending_jobs} pending jobs)")
    return selected


# ==============================================================================
# 2. CIRCUIT RESOURCE ESTIMATOR
# ==============================================================================

def estimate_circuit_resources(
    folder: runner.QuantumBiophysicsFolder,
    backend,
    optimization_level: int = 3,
):
    """
    Transpiles the ansatz circuit to the real backend and reports:
      - Original depth / gate count
      - Transpiled depth
      - Transpiled 2-qubit gate count (cx / ecr / cz)
      - Number of parameters
      - Number of qubits used

    Does NOT execute anything on hardware.
    """
    ansatz = folder.ansatz  # EfficientSU2 circuit

    print(f"\n[INFO] Transpiling circuit to backend '{backend.name}' "
          f"(optimization_level={optimization_level})...")

    t0 = time.time()
    transpiled = transpile(
        ansatz,
        backend=backend,
        optimization_level=optimization_level,
    )
    transpile_time = time.time() - t0

    # ── count 2-qubit gates ────────────────────────────────────────────────────
    TWO_QUBIT_GATES = {"cx", "cz", "ecr", "rzz", "rxx", "ryy", "cp", "swap", "iswap"}
    op_counts = transpiled.count_ops()
    n_2q = sum(v for k, v in op_counts.items() if k.lower() in TWO_QUBIT_GATES)

    results = {
        "backend_name":         backend.name,
        "backend_qubits":       backend.num_qubits,
        "n_qubits_circuit":     folder.n_qubits,
        "n_params":             folder.n_params,
        "total_angles":         folder.total_angles,
        "reps":                 folder.reps,

        # pre-transpilation
        "depth_original":       ansatz.depth(),
        "gate_count_original":  sum(ansatz.count_ops().values()),

        # post-transpilation
        "depth_transpiled":     transpiled.depth(),
        "gate_count_transpiled":sum(op_counts.values()),
        "two_qubit_gates":      n_2q,
        "op_breakdown":         dict(op_counts),

        "transpile_time_s":     round(transpile_time, 3),
        "optimization_level":   optimization_level,
    }
    return results, transpiled


# ==============================================================================
# 3. WALL-CLOCK TIME ESTIMATOR
# ==============================================================================

def estimate_wallclock_time(
    resource_report: dict,
    backend,
    ensemble_size: int,
    shots_per_call: int,
    max_iter: int,
    n_stages: int = 3,
):
    """
    Rough wall-clock estimate for the full folding run on real hardware.

    Key assumptions:
    - Each energy_function call = 3 circuit executions (Z/X/Y bases)
    - Each circuit execution = shots_per_call shots
    - IBM job overhead ≈ 5-15 s per job submission (we use 10 s)
    - Circuit execution time ≈ depth * dt  (dt ~ 0.5 µs per layer on modern hardware)
    - Optimizer calls per stage ≈ max_iter (upper bound)
    - Scouting phase ≈ 50 energy calls before each replica
    """

    # Backend timing parameters
    try:
        dt = backend.dt if hasattr(backend, "dt") and backend.dt else 0.5e-9
    except Exception:
        dt = 0.5e-9   # fallback: 0.5 ns per dt unit

    depth = resource_report["depth_transpiled"]
    shots = shots_per_call

    # Time per single circuit execution on hardware
    # Rough model: circuit_time = depth * dt * shots  + readout overhead
    readout_us_per_shot = 5.0   # µs
    circuit_exec_s = (depth * dt * 1e6 + readout_us_per_shot) * shots * 1e-6

    job_overhead_s = 10.0       # IBM submission + queue overhead per job (best case)

    # Per energy_function call: 3 circuits (Z, X, Y) submitted as 1 job (batched)
    time_per_energy_call_s = 3 * circuit_exec_s + job_overhead_s

    # Optimizer iterations (COBYLA stage 1 + SLSQP stages 2&3)
    # COBYLA is derivative-free: ~1 call per iteration
    # SLSQP uses finite differences: ~n_params+1 calls per iteration (expensive!)
    n_params = resource_report["n_params"]
    calls_stage1 = max_iter                            # COBYLA
    calls_stage2 = max_iter * (n_params + 1)           # SLSQP (worst case)
    calls_stage3 = max_iter * (n_params + 1)           # SLSQP (worst case)
    calls_scouting = 50                                # get_smart_initialization

    total_calls_per_replica = calls_scouting + calls_stage1 + calls_stage2 + calls_stage3

    # Total time
    time_per_replica_s   = total_calls_per_replica * time_per_energy_call_s
    total_time_s         = ensemble_size * time_per_replica_s
    total_time_min       = total_time_s / 60.0
    total_time_hr        = total_time_min / 60.0

    return {
        "shots_per_call":               shots,
        "circuit_exec_s_per_circuit":   round(circuit_exec_s, 4),
        "job_overhead_s":               job_overhead_s,
        "time_per_energy_call_s":       round(time_per_energy_call_s, 4),
        "calls_scouting":               calls_scouting,
        "calls_stage1_cobyla":          calls_stage1,
        "calls_stage2_slsqp_worst":     calls_stage2,
        "calls_stage3_slsqp_worst":     calls_stage3,
        "total_calls_per_replica":      total_calls_per_replica,
        "ensemble_size":                ensemble_size,
        "total_energy_calls":           ensemble_size * total_calls_per_replica,
        "estimated_time_per_replica_hr":round(time_per_replica_hr := time_per_replica_s / 3600, 2),
        "estimated_total_time_min":     round(total_time_min, 1),
        "estimated_total_time_hr":      round(total_time_hr, 2),
        "note": (
            "SLSQP calls scale as O(n_params * max_iter). "
            "Consider switching Stage 2/3 to COBYLA or SPSA for hardware runs."
        ),
    }


# ==============================================================================
# 4. PRETTY PRINTER
# ==============================================================================

def print_report(resource: dict, timing: dict, sequence: str, force_field: str):
    sep = "=" * 65

    print(f"\n{sep}")
    print(f"  QTF HARDWARE RESOURCE REPORT")
    print(f"  Sequence   : {sequence}")
    print(f"  Force field: {force_field.upper()}")
    print(sep)

    print("\n── CIRCUIT (post-transpilation) ──────────────────────────────")
    print(f"  Backend                : {resource['backend_name']} ({resource['backend_qubits']} qubits)")
    print(f"  Circuit qubits used    : {resource['n_qubits_circuit']}")
    print(f"  Ansatz reps            : {resource['reps']}")
    print(f"  Trainable parameters   : {resource['n_params']}")
    print(f"  Total torsion angles   : {resource['total_angles']}")
    print(f"  Depth  (original)      : {resource['depth_original']}")
    print(f"  Depth  (transpiled)    : {resource['depth_transpiled']}")
    print(f"  Gates  (original)      : {resource['gate_count_original']}")
    print(f"  Gates  (transpiled)    : {resource['gate_count_transpiled']}")
    print(f"  2-qubit gates          : {resource['two_qubit_gates']}  ← key hardware cost")
    print(f"  Optimization level     : {resource['optimization_level']}")
    print(f"  Transpile time         : {resource['transpile_time_s']} s")

    print(f"\n  Gate breakdown: {resource['op_breakdown']}")

    print("\n── TIMING ESTIMATE (real hardware) ───────────────────────────")
    print(f"  Shots per circuit call : {timing['shots_per_call']}")
    print(f"  Time per circuit exec  : {timing['circuit_exec_s_per_circuit']:.4f} s")
    print(f"  IBM job overhead       : {timing['job_overhead_s']} s")
    print(f"  Time per energy() call : {timing['time_per_energy_call_s']:.4f} s")
    print(f"  Calls (scouting)       : {timing['calls_scouting']}")
    print(f"  Calls (Stage 1 COBYLA) : {timing['calls_stage1_cobyla']}")
    print(f"  Calls (Stage 2 SLSQP) : {timing['calls_stage2_slsqp_worst']}  [worst case]")
    print(f"  Calls (Stage 3 SLSQP) : {timing['calls_stage3_slsqp_worst']}  [worst case]")
    print(f"  Total calls/replica    : {timing['total_calls_per_replica']:,}")
    print(f"  Total calls (ensemble) : {timing['total_energy_calls']:,}")
    print(f"  Ensemble size          : {timing['ensemble_size']}")
    print(f"\n  ⏱  Est. time/replica   : {timing['estimated_time_per_replica_hr']:.2f} hours")
    print(f"  ⏱  Est. TOTAL runtime  : {timing['estimated_total_time_hr']:.2f} hours "
          f"({timing['estimated_total_time_min']:.0f} min)")

    print(f"\n  ⚠  NOTE: {timing['note']}")

    # Feasibility verdict
    print("\n── FEASIBILITY VERDICT ───────────────────────────────────────")
    total_hr = timing["estimated_total_time_hr"]
    n2q = resource["two_qubit_gates"]

    if total_hr < 2 and n2q < 500:
        verdict = "✅  FEASIBLE  – reasonable depth and runtime for real hardware."
    elif total_hr < 24 and n2q < 2000:
        verdict = "⚠️  MARGINAL  – consider reducing max_iter or using SPSA optimizer."
    else:
        verdict = ("❌  NOT FEASIBLE on real hardware as-is.\n"
                   "     Suggestions:\n"
                   "       1. Reduce --maxiter (try 100-200)\n"
                   "       2. Switch SLSQP stages to COBYLA or SPSA\n"
                   "       3. Reduce --ensemble_size\n"
                   "       4. Use fewer reps in EfficientSU2\n"
                   "       5. Run Stage 1 on simulator, Stage 3 only on hardware")
    print(f"  {verdict}")
    print(sep + "\n")


# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Estimate IBM Quantum hardware resources for QTF without running."
    )
    parser.add_argument("--predict", required=True, help="Amino acid sequence (e.g. YYDPETGTWY)")
    parser.add_argument("--forcefield", default="amber",
                        choices=["amber", "opls", "charmm"],
                        help="Force field (default: amber)")
    parser.add_argument("--chi_mode", default="selective",
                        choices=["chi1_only", "selective", "all"])
    parser.add_argument("--ensemble_size", default=3, type=int)
    parser.add_argument("--maxiter", default=2000, type=int)
    parser.add_argument("--shots", default=4096, type=int,
                        help="Shots per circuit call on hardware (default: 4096)")
    parser.add_argument("--opt_level", default=3, type=int,
                        choices=[0, 1, 2, 3],
                        help="Qiskit transpilation optimization level (default: 3)")

    # Credentials – prefer env vars over CLI for security
    parser.add_argument("--ibm_token", default=None,
                        help="IBM Quantum token (prefer env var QTF_IBM_TOKEN)")
    parser.add_argument("--ibm_instance", default=None,
                        help="IBM CRN instance (prefer env var QTF_IBM_INSTANCE)")
    args = parser.parse_args()

    # ── Credentials ────────────────────────────────────────────────────────────
    token    = args.ibm_token    or os.environ.get("QTF_IBM_TOKEN")
    instance = args.ibm_instance or os.environ.get("QTF_IBM_INSTANCE")

    if not token:
        print("[ERROR] IBM token not found.")
        print("  Set environment variable:  export QTF_IBM_TOKEN='your_token'")
        print("  Or pass via CLI:           --ibm_token 'your_token'")
        sys.exit(1)

    # ── Connect to IBM ─────────────────────────────────────────────────────────
    print("[INFO] Connecting to IBM Quantum...")
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=token,
        instance=instance,
    )

    # ── Build folder (no folding yet, just circuit setup) ──────────────────────
    selective_chi_map = {
        "Y": ["chi1", "chi2"], "W": ["chi1", "chi2"],
        "F": ["chi1", "chi2"], "H": ["chi1", "chi2"],
        "D": ["chi1"], "E": ["chi1"], "N": ["chi1"], "Q": ["chi1"],
        "T": ["chi1"], "S": ["chi1"], "V": ["chi1"], "I": ["chi1"],
        "L": ["chi1"], "M": ["chi1"], "K": ["chi1"], "R": ["chi1"],
        "C": ["chi1"], "P": ["chi1"], "A": [], "G": [],
    }

    print(f"\n[INFO] Building circuit for sequence: {args.predict}")
    folder = runner.QuantumBiophysicsFolder(
        args.predict,
        force_field=args.forcefield,
        chi_mode=args.chi_mode,
        selective_chi_map=selective_chi_map,
    )

    # ── Select least-busy backend ──────────────────────────────────────────────
    backend = get_least_busy_backend(service, min_qubits=folder.n_qubits)

    # ── Transpile & measure resources ─────────────────────────────────────────
    resource_report, _ = estimate_circuit_resources(
        folder, backend, optimization_level=args.opt_level
    )

    # ── Estimate wall-clock time ───────────────────────────────────────────────
    timing_report = estimate_wallclock_time(
        resource_report,
        backend,
        ensemble_size=args.ensemble_size,
        shots_per_call=args.shots,
        max_iter=args.maxiter,
    )

    # ── Print full report ──────────────────────────────────────────────────────
    print_report(resource_report, timing_report, args.predict, args.forcefield)

    # ── Save JSON report ───────────────────────────────────────────────────────
    import json
    report_path = f"hw_estimate_{args.predict}_{args.forcefield}.json"
    with open(report_path, "w") as f:
        json.dump({
            "sequence": args.predict,
            "force_field": args.forcefield,
            "circuit_resources": resource_report,
            "timing_estimate": timing_report,
        }, f, indent=4)
    print(f"[INFO] Report saved to: {report_path}")


if __name__ == "__main__":
    main()