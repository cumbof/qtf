#!/usr/bin/env python3

### general imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import urllib.request
import os, time, json, argparse
from datetime import datetime

### qtf imports
import QTF.evaluator as evaluator
import QTF.runner as runner

def __main__():
    '''
    This function serves as the main entry point for running Quantum Torsion Folder.  This method works by encoding 
    a protein's dihedral/torsion angles into qubit phases, then optimizing these angles using a quantum-inspired optimization 
    algorithm. The final output is a predicted 3D structure of the protein based on the optimized angles.
    
    The optimization is driven by a Hamiltonian that incorporates various energy terms derived from classical force fields,
    such as Amber, OPLS, and CHARMM. These force fields provide the necessary parameters to evaluate the energy of a given conformation, 
    allowing the algorithm to search for low-energy states that correspond to physically plausible protein structures.  
    
    Knowledge of Ramachandran plots and secondary structure propensities is also integrated into the optimization process,
    guiding the search towards conformations that are more likely to be observed in nature. 
      
    The script can operate in two modes: "predict_and_compare" and "predict_only". The former is used to predict the 
    3D structure of a given amino acid sequence and compare it against a reference structure, while the latter focuses 
    solely on predicting the structure without any comparison.
    '''
    
    # start tracking time
    time_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # parse command line arguments
    # example usage: python qtf_predictor.py --predict "YYDPETGTWY" --reference_structure "5AWL" --average_reference_backbone False --forcefield "amber" --mode "predict_and_compare" --ensemble_size 3 --prime_strategy "Random"
    parser = argparse.ArgumentParser()

    parser.add_argument('--predict', default=None, help='target sequence to predict')

    parser.add_argument('--reference_structure', default=None, help='reference structure PDB ID for comparison')
    
    parser.add_argument('--average_reference_backbone', 
                        default=False,
                        type=bool,
                        help='How to select backbone from reference structure, either first model or average for NMR ensembles. Defaults to first model, which automatically works with Xray structures.')
    
    parser.add_argument('--forcefield', default="amber", choices=["amber", "opls", "charmm", "all"], help='choice of force field for scoring')
    
    parser.add_argument('--mode', default="predict_and_compare", choices=["predict_and_compare", "predict_only"], help='which mode to run script in')

    parser.add_argument('--ensemble_size', default=3, type=int, help='ensemble size')
    
    parser.add_argument('--prime_strategy', default="Random", choices=["Random", "mixed", "Helix", "Sheet"], help='prime strategy for initialization')

    args=parser.parse_args()

    # set the arguments that are passed in, which can then be applied to both modes
    force_field = args.forcefield
    reference_structure_pdb_id = args.reference_structure
    average_reference_backbone_mode = args.average_reference_backbone
    sequence = args.predict
    ensemble_size = args.ensemble_size
    prime_strategy = args.prime_strategy

    # set the output directory and change into it
    os.mkdir("outputs") if not os.path.exists("outputs") else None
    job_output_dir = os.mkdir("outputs" + f"/{sequence}_{force_field}_{timestamp}")
    if job_output_dir is not None:
        # Create the directory if it doesn't exist (including any necessary parent directories)
        os.makedirs(job_output_dir, exist_ok=True)
        # Change the current working directory
        os.chdir(job_output_dir)
        print(f"Changed directory to: {os.getcwd()}")
    else:
        print("Error: job_output_dir variable is None. Cannot change directory.")
        # Handle the error or exit the script gracefully

    # run the appropriate mode
    if args.mode == "predict_and_compare":
        # 1. Setup Chignolin Sequence
        print(f"--- DIAGNOSING BACKBONE: {sequence} ---")

        # 2. Initialize Folder & Manager
        if force_field=="all":
            folder = runner.QuantumBiophysicsFolder(sequence, force_field=[ff for ff in list("amber", "opls", "charmm")])
        else:
            folder = runner.QuantumBiophysicsFolder(sequence, force_field=force_field)
        manager = runner.EnsembleFoldingManager(folder)

        # 3. Run Ensemble (Using the Smart Initialization) 
        # We run 3 replicas with mixed strategies (Helix, Sheet, Random)
        manager.run_ensemble(n_runs=ensemble_size, prime_strategy=prime_strategy)

        # 4. Get Best Result
        best_result = manager.evaluate_best()
        final_coords = best_result['coords']
        tracker = best_result['tracker']

        # 5. Extract Backbone (CA) for Validation
        # Filter labels where atom name is 'CA'
        pred_ca = np.array([final_coords[i] for i, lbl in enumerate(folder.static_labels) if lbl[1] == 'CA'])
        true_ca = evaluator.get_ground_truth_backbone(reference_structure_pdb_id, average_reference_backbone_mode)

        # Truncate to match lengths (in case of differing caps)
        n = min(len(pred_ca), len(true_ca))
        pred_ca = pred_ca[:n]; true_ca = true_ca[:n]

        # 6. Calculate Metrics
        p_e2e, p_rg = evaluator.calculate_physics_metrics(pred_ca)
        t_e2e, t_rg = evaluator.calculate_physics_metrics(true_ca)

        print(f"Ensemble size is {ensemble_size}")
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
        #plt.show()
        plt.savefig(f"Backbone_Diagnosis_{sequence}_{force_field}_{timestamp}.png", dpi=600)

    elif args.mode == "predict_only":
        # 1. Setup Chignolin Sequence
        print(f"--- PREDICTING BACKBONE: {sequence} ---")

        # 2. Initialize Folder & Manager
        folder = runner.QuantumBiophysicsFolder(sequence, force_field=force_field)
        manager = runner.EnsembleFoldingManager(folder)

        # 3. Run Ensemble (Using the Smart Initialization) 
        # We run 3 replicas with mixed strategies (Helix, Sheet, Random)
        manager.run_ensemble(n_runs=ensemble_size, prime_strategy=prime_strategy)

        # 4. Get Best Result
        best_result = manager.evaluate_best()
        final_coords = best_result['coords']
        tracker = best_result['tracker']

        print(f"\nPrediction Complete. Best run ID: {best_result['id']}")

    time_end = time.time()
    runtime = f"{((time_end - time_start) / 60):.2f}"
    print(f"Total execution time for generating models: {runtime} minutes")
    # --- build a single record for THIS run ---
    summary_data = {
        # metrics
        "End-to-End Dist (Å)": p_e2e,
        "End-to-End Target (Å)": t_e2e,
        "End-to-End Status": "EXPANDED" if p_e2e > t_e2e + 5 else "GOOD",

        "Rg (Å)": p_rg,
        "Rg Target (Å)": t_rg,
        "Rg Status": "PUFFY" if p_rg > t_rg + 2 else "COMPACT",

        "Backbone RMSD (Å)": float(f"{rmsd:.3f}"),
        "RMSD Status": "GOOD" if rmsd < 2.0 else "BAD",

        # run-level meta/settings
        "Runtime (minutes)": runtime,
        "Ensemble Size": ensemble_size,
        "Sequence": sequence,
        "mode": args.mode,
        "Reference Structure": (reference_structure_pdb_id if reference_structure_pdb_id is not None and args.mode!="predict_only" else None),
        "Force Field": force_field,
        "Prime Strategy": prime_strategy,
    }
    df = pd.DataFrame([summary_data])
    df.to_csv(os.path.join(job_output_dir, "summary_results.csv"), index=False)

    with open(os.path.join(job_output_dir, "summary_results.json"), "w") as f:
        json.dump(summary_data, f, indent=4)


    ## now we can start appending an onto a master results file in the outputs directory    
    # first, append the master dataframe
    master_csv_path = "outputs/master_summary_results.csv"
    
    try:
        df_all = pd.read_csv(master_csv_path)
        df_all = pd.concat([df_all, df], ignore_index=True)
    except FileNotFoundError:
        df_all = df

    df_all.to_csv(master_csv_path, index=False)
    
    # next, append the master json file 
    master_json_path = "outputs/master_summary_results.jsonl"
    with open(master_json_path, "a") as f:
        f.write(json.dumps(summary_data) + "\n")

if __name__ == "__main__":
    __main__()  
