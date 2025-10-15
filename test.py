from smavnet_sim import SimulationConfig, SmavNet2D

cfg = SimulationConfig(
    n_agents=18,
    comm_r=100.0,
    duration=900.0,
    speed=10.0,
    seed=42,             # optional for determinism
    user_mode="uniform", # fixed|uniform|gaussian
)
sim = SmavNet2D(cfg)
# Headless fast run to the configured duration (records first success time)
#result = sim.run(headless=True)
#print(result.success, result.success_time, result.user_pos)
# Visual run
sim.reset(seed=42)  
result_viz = sim.run(headless=False)