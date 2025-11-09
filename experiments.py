import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from smavnet_sim_final import SimulationConfig, SmavNet2D


def run_trials_for_size(n_agents: int, num_trials: int, duration_s: float, base_seed: int = 12345) -> float:
    successes = 0
    for k in tqdm(range(num_trials), desc=f"Trials ({n_agents} MAVs)", leave=False):
        # stagger seed to vary user placement and randomness
        seed = base_seed + n_agents * 100000 + k
        cfg = SimulationConfig(
            n_agents=n_agents,
            duration=duration_s,
            seed=seed,
            user_mode="uniform",
            area_w = 800.0,
            area_h = 600.0,
            terminate_on_success=True,
        )
        sim = SmavNet2D(cfg)
        result = sim.run(headless=True)
        if result.success and (result.success_time is not None) and (result.success_time <= duration_s):
            successes += 1
    return successes / float(num_trials)


def _rotate(vec: np.ndarray, degrees: float) -> np.ndarray:
    th = np.deg2rad(degrees)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return R.dot(vec)


def _generate_lattice_nodes(area_w: float, area_h: float, base_pos: np.ndarray,
                            comm_r: float, step_factor: float = 0.95, angle_deg: float = 30.0) -> np.ndarray:
    base_dir = np.array([0.0, 1.0])
    v_left = _rotate(base_dir, +angle_deg)
    v_right = _rotate(base_dir, -angle_deg)
    step_dist = comm_r * step_factor

    positions = []
    # iterate i,j in non-negative quadrant until out of bounds
    # conservative max counts based on area diagonal
    max_steps = int(2 * max(area_w, area_h) / (step_dist * np.cos(np.deg2rad(angle_deg)) + 1e-9)) + 2
    for i in range(0, max_steps):
        for j in range(0, max_steps):
            pos = base_pos + step_dist * (i * v_left + j * v_right)
            if 0 <= pos[0] <= area_w and 0 <= pos[1] <= area_h:
                positions.append(pos)
    if not positions:
        return np.zeros((0, 2))
    return np.vstack(positions)


def _coverage_mask(area_w: float, area_h: float, nodes_xy: np.ndarray, radius: float, grid_res: float = 5.0) -> np.ndarray:
    gx = int(np.ceil(area_w / grid_res))
    gy = int(np.ceil(area_h / grid_res))
    mask = np.zeros((gy, gx), dtype=bool)
    if nodes_xy.size == 0:
        return mask
    yy, xx = np.mgrid[0:gy, 0:gx]
    xs = (xx + 0.5) * grid_res
    ys = (yy + 0.5) * grid_res
    for (x0, y0) in nodes_xy:
        dist2 = (xs - x0) ** 2 + (ys - y0) ** 2
        mask |= dist2 <= (radius ** 2)
    return mask


def run_instantaneous_success_prob_experiment(
    n_agents: int,
    num_trials: int,
    max_duration_s: float,
    area_w: float,
    area_h: float,
    comm_r: float,
    step_factor: float,
    angle_deg: float,
    base_seed: int = 12345
) -> None:
    """Run experiments to measure instantaneous success probability at each 5 seconds."""
    print(f"Running instantaneous success probability experiment for {n_agents} MAVs...")
    
    # Time points to evaluate (every 5 seconds)
    time_points = np.arange(0, max_duration_s + 1, 5)  # Every 5 seconds
    success_probs = []
    
    for t in tqdm(time_points, desc=f"Evaluating every 5 seconds for {n_agents} MAVs"):
        successes = 0
        for k in range(num_trials):
            seed = base_seed + n_agents * 100000 + k + int(t)  # Include time in seed for variation
            cfg = SimulationConfig(
                n_agents=n_agents,
                duration=t,
                seed=seed,
                user_mode="uniform",
                area_w=area_w,
                area_h=area_h,
                terminate_on_success=True,
            )
            sim = SmavNet2D(cfg)
            result = sim.run(headless=True)
            if result.success and (result.success_time is not None) and (result.success_time <= t):
                successes += 1
        success_probs.append(successes / float(num_trials))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(time_points / 60.0, success_probs, linewidth=2)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Instantaneous Success Probability')
    plt.title(f'Instantaneous Success Probability vs Time (N_MAVs={n_agents}, {num_trials} trials/sec)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_duration_s / 60.0)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    out_file = f'instantaneous_success_prob_n{n_agents}_trials{num_trials}.png'
    plt.savefig(out_file, dpi=1000)
    plt.close()
    print(f"Saved plot to {out_file}")


def plot_user_distribution_with_coverage(n_agents: int, num_trials: int, duration_s: float,
                                         area_w: float = 1000.0, area_h: float = 1000.0,
                                         comm_r: float = 100.0, step_factor: float = 0.95,
                                         angle_deg: float = 30.0, base_seed: int = 12345) -> None:
    base_pos = np.array([area_w / 2.0, 0.0])

    users_success = []
    users_fail = []

    for k in tqdm(range(num_trials), desc=f"User trials ({n_agents} MAVs)"):
        seed = base_seed + n_agents * 100000 + k
        cfg = SimulationConfig(
            n_agents=n_agents,
            duration=duration_s,
            seed=seed,
            user_mode="uniform",
            area_w=area_w,
            area_h=area_h,
            comm_r=comm_r,
            terminate_on_success=True,
        )
        sim = SmavNet2D(cfg)
        result = sim.run(headless=True)
        if result.success and (result.success_time is not None) and (result.success_time <= duration_s):
            users_success.append(result.user_pos)
        else:
            users_fail.append(result.user_pos)

    users_success = np.array(users_success) if users_success else np.zeros((0, 2))
    users_fail = np.array(users_fail) if users_fail else np.zeros((0, 2))

    # lattice nodes and coverage union mask
    nodes_xy = _generate_lattice_nodes(area_w, area_h, base_pos, comm_r, step_factor, angle_deg)
    mask = _coverage_mask(area_w, area_h, nodes_xy, radius=comm_r, grid_res=3.0)

    plt.figure(figsize=(10, 8))
    # coverage: show a single uniform transparency so overlaps don't darken
    plt.imshow(mask, cmap=plt.cm.Greys, interpolation='nearest', alpha=0.18,
               extent=[0, area_w, 0, area_h], origin='lower')

    # draw lattice node centers (optional, faint)
    if nodes_xy.size:
        plt.scatter(nodes_xy[:, 0], nodes_xy[:, 1], s=8, c='0.6', alpha=0.6, linewidths=0)

    # base and users
    plt.scatter([base_pos[0]], [base_pos[1]], c='black', s=120, marker='s', label='Base')
    if users_success.size:
        plt.scatter(users_success[:, 0], users_success[:, 1], c='green', s=18, marker='o', label='Success users', alpha=0.8)
    if users_fail.size:
        plt.scatter(users_fail[:, 0], users_fail[:, 1], c='red', s=24, marker='x', label='Failed users', alpha=0.8)

    plt.xlim(0, area_w)
    plt.ylim(-50.0, area_h)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"User distribution and coverage (n_agents={n_agents}, trials={num_trials})")
    plt.legend(loc='lower right', framealpha=0.9)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    out_file = f'user_distribution_coverage_n{n_agents}_trials{num_trials}.png'
    plt.savefig(out_file, dpi=1000)
    plt.close()


def run_scalability_experiment(
    swarm_sizes: List[int],
    num_trials: int,
    duration_s: float,
    area_w: float,
    area_h: float,
    comm_r: float,
    step_factor: float,
    angle_deg: float,
    base_seed: int = 12345
) -> None:
    """Run scalability experiment and plot success probability vs swarm size."""
    print("Running scalability experiment...")
    probs: List[float] = []

    t0 = time.time()
    for n in tqdm(swarm_sizes, desc="Swarm sizes"):
        p = run_trials_for_size(n, num_trials, duration_s)
        probs.append(p)
        print(f"n_agents={n}: success_prob={p:.3f}")
    t1 = time.time()
    print(f"Completed scalability experiment in {(t1 - t0)/60.0:.2f} min")

    # Plot scalability results
    plt.figure(figsize=(7, 4))
    plt.plot(swarm_sizes, probs, marker='o')
    plt.xlabel('Number of MAVs')
    plt.ylabel('Probability of finding user in 30 min')
    plt.title('SMAVNET Scalability: Success Probability vs Swarm Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scalability_success_prob.png', dpi=1000)
    plt.close()
    print("Saved scalability plot to scalability_success_prob.png")


def main() -> None:
    # Experiment parameters
    swarm_sizes: List[int] = list(range(5, 21))
    num_trials = 500
    duration_s = 30.0 * 60.0
    area_w = 800.0
    area_h = 600.0
    comm_r = 100.0
    step_factor = 0.95
    angle_deg = 30.0
    base_seed = 12345


    # Generate user distribution and coverage plot
    plot_user_distribution_with_coverage(
        n_agents=15,
        num_trials=num_trials,
        duration_s=duration_s,
        area_w=area_w,
        area_h=area_h,
        comm_r=comm_r,
        step_factor=step_factor,
        angle_deg=angle_deg,
        base_seed=base_seed
    )

    # Run scalability experiment
    run_scalability_experiment(
        swarm_sizes=swarm_sizes,
        num_trials=num_trials,
        duration_s=duration_s,
        area_w=area_w,
        area_h=area_h,
        comm_r=comm_r,
        step_factor=step_factor,
        angle_deg=angle_deg,
        base_seed=base_seed
    )

    # Run instantaneous success probability experiment for a sample swarm size
    run_instantaneous_success_prob_experiment(
        n_agents=15,
        num_trials=500,  # Fixed 500 trials per 5 seconds
        max_duration_s=duration_s,
        area_w=area_w,
        area_h=area_h,
        comm_r=comm_r,
        step_factor=step_factor,
        angle_deg=angle_deg,
        base_seed=base_seed
    )

if __name__ == "__main__":
    main()


