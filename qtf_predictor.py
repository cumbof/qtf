### general imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import urllib.request
import os
import time

### qtf imports
import QTF.evaluator as evaluator
import QTF.runner as runner

def __main__():
    time_start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--predict', default=None, help='target sequence to predict')

    parser.add_argument('--reference', default=None, help='reference structure PDB ID for comparison')
    
    parser.add_argument('--forcefield', default="amber", choices=list("amber", "opls", "charmm", "all"), help='choice of force field for scoring')
    
    parser.add_argument('--mode', default="predict_and_compare", choices=list("predict_and_compare", "predict_only"), help='which mode to run script in')

    parser.add_argument('--ensemble_size', default=3, type=int, help='ensemble size')
    
    parser.add_argument('--prime_strategy', default="Random", choices=list("Random", "mixed", "Helix", "Sheet"), help='prime strategy for initialization')

    args=parser.parse_args()


    if args.mode == "predict_and_compare":
        # 1. Setup Chignolin Sequence
        sequence = args.predict 
        print(f"--- DIAGNOSING BACKBONE: {sequence} ---")

        # 2. Initialize Folder & Manager
        if args.forcefield=="all":
            folder = runner.QuantumBiophysicsFolder(sequence, force_field=[ff for ff in list("amber", "opls", "charmm")])
        else:
            folder = runner.QuantumBiophysicsFolder(sequence, force_field=args.forcefield)
        manager = runner.EnsembleFoldingManager(folder)

        # 3. Run Ensemble (Using the Smart Initialization) 
        # We run 3 replicas with mixed strategies (Helix, Sheet, Random)
        manager.run_ensemble(n_runs=args.ensemble_size, prime_strategy=args.prime_strategy)

        # 4. Get Best Result
        best_result = manager.evaluate_best()
        final_coords = best_result['coords']
        tracker = best_result['tracker']

        # 5. Extract Backbone (CA) for Validation
        # Filter labels where atom name is 'CA'
        reference_structure_pdb_id = args.reference
        pred_ca = np.array([final_coords[i] for i, lbl in enumerate(folder.static_labels) if lbl[1] == 'CA'])
        true_ca = evaluator.get_ground_truth_backbone(reference_structure_pdb_id)

        # Truncate to match lengths (in case of differing caps)
        n = min(len(pred_ca), len(true_ca))
        pred_ca = pred_ca[:n]; true_ca = true_ca[:n]

        # 6. Calculate Metrics
        p_e2e, p_rg = evaluator.calculate_physics_metrics(pred_ca)
        t_e2e, t_rg = evaluator.calculate_physics_metrics(true_ca)

        print(f"\nMetric             | Predicted | Target (5AWL) | Status")
        print(f"-------------------|-----------|---------------|-------")
        print(f"End-to-End Dist    | {p_e2e:6.2f} Å | {t_e2e:6.2f} Å      | {'EXPANDED' if p_e2e > t_e2e + 5 else 'GOOD'}")
        print(f"Radius of Gyration | {p_rg:6.2f} Å | {t_rg:6.2f} Å      | {'PUFFY' if p_rg > t_rg + 2 else 'COMPACT'}")

        # 7. RMSD & Dual Plots
        rmsd, aligned_pred = evaluator.kabsch_backbone_align(pred_ca, true_ca)
        print(f"\nBackbone RMSD: {rmsd:.3f} Å")

        fig = plt.figure(figsize=(16, 7))

        # PLOT 1: 3D Structure Comparison 
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(aligned_pred[:,0], aligned_pred[:,1], aligned_pred[:,2], '-o', c='blue', label='Prediction (Best)', lw=2)
        ax1.plot(true_ca[:,0], true_ca[:,1], true_ca[:,2], '--o', c='red', label='Target (5AWL)', alpha=0.7)
        ax1.set_title(f"3D Structure Alignment\nRMSD: {rmsd:.2f} Å", fontsize=12, fontweight='bold')
        ax1.legend()

        # PLOT 2: Energy Landscape of the Winner
        ax2 = fig.add_subplot(122)
        energies = np.array(tracker.history)
        energies = np.clip(energies, -1000, 2000) # Clip visuals
        ax2.plot(energies, color='#2c3e50', lw=1.5, label='Hamiltonian Energy')

        colors = ['#e74c3c', '#f1c40f', '#0f26f1']
        for i, (idx, name) in enumerate(tracker.stage_markers):
            if idx < len(energies):
                ax2.axvline(x=idx, color=colors[i], linestyle='--', alpha=0.8)
                ax2.text(idx + 10, max(energies)*0.9, name, color=colors[i], fontweight='bold')
                
        ax2.set_title(f"Optimization Landscape (Run #{best_result['id']})", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Evaluations")
        ax2.set_ylabel("Energy")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    elif args.mode == "predict_only":
        # 1. Setup Chignolin Sequence
        sequence = args.predict 
        print(f"--- PREDICTING BACKBONE: {sequence} ---")

        # 2. Initialize Folder & Manager
        folder = runner.QuantumBiophysicsFolder(sequence, force_field=args.forcefield)
        manager = runner.EnsembleFoldingManager(folder)

        # 3. Run Ensemble (Using the Smart Initialization) 
        # We run 3 replicas with mixed strategies (Helix, Sheet, Random)
        manager.run_ensemble(n_runs=args.ensemble_size, prime_strategy=args.prime_strategy)

        # 4. Get Best Result
        best_result = manager.evaluate_best()
        final_coords = best_result['coords']
        tracker = best_result['tracker']

        print(f"\nPrediction Complete. Best run ID: {best_result['id']}")

    time_end = time.time()
    print(f"Total Execution Time: {time_end - time_start:.2f} seconds")

if __name__ == "__main__":
    __main__()  
