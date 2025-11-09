import time
import os
from datetime import datetime
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from smavnet_sim_final_wind import SimulationConfig, SmavNet2D


def get_output_dir() -> str:
    """Create and return a timestamped output directory for plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"replacement_drift_experiments_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def run_trials_for_size_with_metrics(n_agents: int, num_trials: int, duration_s: float,
                                     enable_wind: bool = True, enable_replacement: bool = True,
                                     wind_speed: float = 2.0, base_seed: int = 12345):
    """Run trials and collect drift and swap metrics."""
    successes = 0
    avg_drifts = []
    avg_drift_reductions = []
    num_swaps_list = []
    
    for k in tqdm(range(num_trials), desc=f"Trials with metrics ({n_agents} MAVs)", leave=False):
        seed = base_seed + n_agents * 100000 + k
        cfg = SimulationConfig(
            n_agents=n_agents,
            duration=duration_s,
            seed=seed,
            user_mode="uniform",
            area_w=800.0,
            area_h=600.0,
            enable_wind=enable_wind,
            wind_speed=wind_speed,
            enable_replacement=enable_replacement,
            terminate_on_success=True,
            log=False,
        )
        sim = SmavNet2D(cfg)
        result = sim.run(headless=True)
        if result.success and (result.success_time is not None) and (result.success_time <= duration_s):
            successes += 1
        avg_drifts.append(result.avg_drift_raw)
        if result.num_swaps > 0:
            avg_drift_reductions.append(result.avg_drift_reduction_raw)
        num_swaps_list.append(result.num_swaps)
    
    return {
        'success_prob': successes / float(num_trials),
        'avg_drift': np.mean(avg_drifts) if avg_drifts else 0.0,
        'std_drift': np.std(avg_drifts) if avg_drifts else 0.0,
        'avg_drift_reduction': np.mean(avg_drift_reductions) if avg_drift_reductions else 0.0,
        'avg_num_swaps': np.mean(num_swaps_list),
        'total_swaps': sum(num_swaps_list),
    }

def run_replacement_comparison_experiment(
    swarm_sizes: List[int],
    num_trials: int,
    duration_s: float,
    area_w: float,
    area_h: float,
    comm_r: float,
    wind_speed: float = 2.0,
    base_seed: int = 12345,
    output_dir: str = "."
) -> None:
    """Compare drift metrics with and without replacement."""
    print("Running replacement comparison experiment...")
    
    drifts_with_rep = []
    drifts_without_rep = []
    swaps_list = []
    
    t0 = time.time()
    for n in tqdm(swarm_sizes, desc="Swarm sizes"):
        # With replacement
        metrics_with = run_trials_for_size_with_metrics(
            n, num_trials, duration_s, enable_wind=True, enable_replacement=True, 
            wind_speed=wind_speed, base_seed=base_seed
        )
        drifts_with_rep.append(metrics_with['avg_drift'])
        swaps_list.append(metrics_with['avg_num_swaps'])
        
        # Without replacement
        metrics_without = run_trials_for_size_with_metrics(
            n, num_trials, duration_s, enable_wind=True, enable_replacement=False,
            wind_speed=wind_speed, base_seed=base_seed + 50000
        )
        drifts_without_rep.append(metrics_without['avg_drift'])
        
        print(f"n_agents={n}: with_rep: drift={metrics_with['avg_drift']:.2f}m, swaps={metrics_with['avg_num_swaps']:.1f}")
        print(f"          without_rep: drift={metrics_without['avg_drift']:.2f}m")
    
    t1 = time.time()
    print(f"Completed replacement comparison experiment in {(t1 - t0)/60.0:.2f} min")
    
    # Plot drift comparison
    plt.figure(figsize=(10, 6))
    plt.plot(swarm_sizes, drifts_with_rep, marker='o', label='With Replacement', linewidth=2)
    plt.plot(swarm_sizes, drifts_without_rep, marker='s', label='Without Replacement', linewidth=2)
    plt.xlabel('Number of MAVs')
    plt.ylabel('Average Drift (m)')
    plt.title(f'Average Drift vs Swarm Size (Wind={wind_speed}m/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file = os.path.join(output_dir, 'replacement_comparison_drift.png')
    plt.savefig(out_file, dpi=1000)
    plt.close()
    print(f"Saved drift comparison plot to {out_file}")
    
    # Plot number of swaps
    plt.figure(figsize=(10, 6))
    plt.plot(swarm_sizes, swaps_list, marker='o', linewidth=2, color='green')
    plt.xlabel('Number of MAVs')
    plt.ylabel('Average Number of Swaps')
    plt.title(f'Number of Swaps vs Swarm Size (Wind={wind_speed}m/s)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file = os.path.join(output_dir, 'replacement_swaps_vs_swarm_size.png')
    plt.savefig(out_file, dpi=1000)
    plt.close()
    print(f"Saved swaps plot to {out_file}")


def run_drift_vs_replacement_experiment(
    n_agents: int,
    num_trials: int,
    duration_s: float,
    wind_speed: float = 2.0,
    base_seed: int = 12345,
    output_dir: str = "."
) -> None:
    """Run experiment to analyze drift reduction effectiveness."""
    print(f"Running drift vs replacement experiment for {n_agents} MAVs...")
    
    all_pre_drifts = []
    all_post_drifts = []
    all_reductions = []
    all_swap_times = []
    all_swap_ages = []
    
    for k in tqdm(range(num_trials), desc=f"Collecting swap data ({n_agents} MAVs)"):
        seed = base_seed + n_agents * 100000 + k
        cfg = SimulationConfig(
            n_agents=n_agents,
            duration=duration_s,
            seed=seed,
            user_mode="uniform",
            area_w=800.0,
            area_h=600.0,
            enable_wind=True,
            wind_speed=wind_speed,
            enable_replacement=True,
            terminate_on_success=True,
            log=False,
        )
        sim = SmavNet2D(cfg)
        result = sim.run(headless=True)
        
        if result.num_swaps > 0:
            all_pre_drifts.extend(result.swap_pre_drifts)
            all_post_drifts.extend(result.swap_post_drifts)
            all_reductions.extend(result.swap_drift_reductions)
            all_swap_times.extend(result.swap_times)
            all_swap_ages.extend(result.swap_ages)
    
    if len(all_reductions) == 0:
        print("No swaps occurred in any trial")
        return
    
    # Plot pre vs post drift
    plt.figure(figsize=(10, 6))
    plt.scatter(all_pre_drifts, all_post_drifts, alpha=0.5, s=20)
    plt.plot([0, max(all_pre_drifts)], [0, max(all_pre_drifts)], 'r--', label='No reduction', linewidth=1)
    plt.xlabel('Pre-Swap Drift (m)')
    plt.ylabel('Post-Swap Drift (m)')
    plt.title(f'Swap Effectiveness: Pre vs Post Drift (N_MAVs={n_agents}, {num_trials} trials)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file = os.path.join(output_dir, f'replacement_pre_post_drift_n{n_agents}_trials{num_trials}.png')
    plt.savefig(out_file, dpi=1000)
    plt.close()
    print(f"Saved pre/post drift plot to {out_file}")
    
    # Plot drift reduction vs pre-swap drift
    plt.figure(figsize=(10, 6))
    plt.scatter(all_pre_drifts, all_reductions, alpha=0.5, s=20)
    plt.xlabel('Pre-Swap Drift (m)')
    plt.ylabel('Drift Reduction (m)')
    plt.title(f'Drift Reduction vs Pre-Swap Drift (N_MAVs={n_agents}, {num_trials} trials)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file = os.path.join(output_dir, f'replacement_reduction_vs_pre_drift_n{n_agents}_trials{num_trials}.png')
    plt.savefig(out_file, dpi=1000)
    plt.close()
    print(f"Saved reduction plot to {out_file}")
    
    # Plot drift reduction over time
    if len(all_swap_times) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(all_swap_times, all_reductions, alpha=0.5, s=20)
        plt.xlabel('Time (s)')
        plt.ylabel('Drift Reduction (m)')
        plt.title(f'Drift Reduction Over Time (N_MAVs={n_agents}, {num_trials} trials)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_file = os.path.join(output_dir, f'replacement_reduction_over_time_n{n_agents}_trials{num_trials}.png')
        plt.savefig(out_file, dpi=1000)
        plt.close()
        print(f"Saved time plot to {out_file}")
    
    # Plot drift reduction vs node age
    if len(all_swap_ages) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(all_swap_ages, all_reductions, alpha=0.5, s=20)
        plt.xlabel('Node Age at Swap (s)')
        plt.ylabel('Drift Reduction (m)')
        plt.title(f'Drift Reduction vs Node Age (N_MAVs={n_agents}, {num_trials} trials)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_file = os.path.join(output_dir, f'replacement_reduction_vs_age_n{n_agents}_trials{num_trials}.png')
        plt.savefig(out_file, dpi=1000)
        plt.close()
        print(f"Saved age plot to {out_file}")
    
    print(f"Total swaps analyzed: {len(all_reductions)}")
    print(f"Average reduction: {np.mean(all_reductions):.2f}m (std: {np.std(all_reductions):.2f}m)")


def main() -> None:
    # Create output directory with timestamp
    output_dir = get_output_dir()
    print(f"All plots will be saved to: {output_dir}\n")
    
    # Experiment parameters
    swarm_sizes: List[int] = list(range(5, 21))
    num_trials = 500
    duration_s = 30.0 * 60.0
    area_w = 800.0
    area_h = 600.0
    comm_r = 100.0
    step_factor = 0.95
    angle_deg = 30.0
    wind_speed = 2.0
    base_seed = 12345

    # Run replacement comparison experiment (with vs without replacement)
    run_replacement_comparison_experiment(
        swarm_sizes=swarm_sizes,
        num_trials=num_trials,
        duration_s=duration_s,
        area_w=area_w,
        area_h=area_h,
        comm_r=comm_r,
        wind_speed=wind_speed,
        base_seed=base_seed + 500000,
        output_dir=output_dir
    )
    
    # Run drift vs replacement analysis for a sample swarm size
    run_drift_vs_replacement_experiment(
        n_agents=15,
        num_trials=num_trials,
        duration_s=duration_s,
        wind_speed=wind_speed,
        base_seed=base_seed + 700000,
        output_dir=output_dir
    )
    print(f"\nAll experiments completed! Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()


