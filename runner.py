import pandas as pd
from simulation import Simulation
import json
import datetime
import numpy as np
import time
import os

# --- JSON ENCODER FIX ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_user_input(prompt, default, type_func):
    val = input(f"{prompt} [Default: {default}]: ")
    return type_func(val) if val else default

def run_monte_carlo():
    print("\n" + "="*50)
    print("      ALCHIFLY: MONTE CARLO SIMULATION      ")
    print("="*50)
    
    n_friend = get_user_input("No. of Friendly Drones", 8, int)
    n_host_g = get_user_input("No. of Hostile Ground", 4, int)
    n_host_a = get_user_input("No. of Hostile Air", 3, int)
    n_iter = get_user_input("Number of Iterations", 1, int)
    
    results_history = []
    batch_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n[SYSTEM] Starting {n_iter} Iterations...")

    for i in range(n_iter):
        # 1. Run Simulation
        config = {
            "name": f"Run_{i+1}",
            "num_friendlies": n_friend,
            "num_hostiles_ground": n_host_g,
            "num_hostiles_air": n_host_a,
            "asset_patrol_radius": 2000.0,
            "communication_enabled": True
        }
        
        sim = Simulation(config)
        while sim.is_running:
            sim.run_step()
            
        res = sim.get_results()
        
        # 2. PRINT EXACT OUTPUT FORMAT
        # Define the exact order of keys you requested
        display_keys = [
            "Scenario",
            "Time_Taken",
            "Mission_Success",
            "Asset_Damage_Avg",
            "Drones_Lost",
            "Hostiles_Neutralized",
            "Survivors_on_Field",
            "Total_Ammo_Fired",
            "RTB_Damage",
            "RTB_Fuel",
            "RTB_Ammo"
        ]

        print("") # Spacing line
        for key in display_keys:
            # Format: Key (Left aligned 30 chars) Value
            val = res.get(key, "N/A")
            print(f"{key:<30} {val}")
        print("") # Spacing line

        # 3. Save Logs
        path_logs = res.pop("Path_Logs")
        battle_logs = res.pop("Battle_Logs")
        
        # Flatten Path Logs
        flat_paths = []
        for did, entries in path_logs.items():
            for e in entries:
                e['Drone_ID'] = did
                flat_paths.append(e)
        
        # Save JSONs
        with open(f"alchifly_paths_{batch_ts}_iter_{i+1}.json", "w") as f:
            json.dump(flat_paths, f, cls=NumpyEncoder)
            
        with open(f"alchifly_events_{batch_ts}_iter_{i+1}.json", "w") as f:
            json.dump(battle_logs, f, cls=NumpyEncoder, indent=2)

        results_history.append(res)

    # 4. Final Summary
    success_count = sum(1 for r in results_history if r['Mission_Success'])
    print("="*50)
    print(f"Final Summary: {success_count}/{n_iter} Successful Runs")
    print("="*50)

if __name__ == "__main__":
    run_monte_carlo()