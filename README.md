# SMAVNET 2D Simulator

A Python implementation of the Swarm-based Micro Air Vehicle Network (SMAVNET) system for communication relay in 2D environments. This project recreates the ant-based swarming algorithm described in "Ant-based swarming with positionless micro air vehicles for communication relay" adapted for 2D simulation.

## Authors

- **Mayank Chandak** (ME22B224)
- **Aldis Daniel** (ME22B070)

## Overview

SMAVNET is a bio-inspired communication relay system that uses Micro Air Vehicles (MAVs) to establish and maintain communication networks. The system employs ant colony optimization principles where MAVs act as mobile communication nodes, creating a dynamic network that can adapt to changing conditions and user locations.

### Key Features

- **Bio-inspired Algorithm**: Implements pheromone-based path selection similar to ant colony optimization
- **Dynamic Network Formation**: MAVs autonomously create and maintain communication relay networks
- **User Search Capability**: System can locate and establish communication with users in the search area
- **Configurable Parameters**: Extensive configuration options for simulation parameters
- **Headless and Visual Modes**: Fast headless simulation for experiments and optional real-time visualization
- **Comprehensive Experiments**: Built-in scalability and performance analysis tools

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from smavnet_sim_final import SimulationConfig, SmavNet2D

# Create simulation configuration
cfg = SimulationConfig(
    n_agents=18,           # Number of MAVs
    comm_r=100.0,          # Communication range (meters)
    duration=900.0,        # Simulation duration (seconds)
    speed=10.0,            # MAV speed (m/s)
    area_w=800.0,          # Area width (meters)
    area_h=600.0,          # Area height (meters)
    seed=42                # Random seed for reproducibility
)

# Create and run simulation
sim = SmavNet2D(cfg)
result = sim.run(headless=True)

# Check results
print(f"Success: {result.success}")
print(f"Success Time: {result.success_time}s")
print(f"User Position: {result.user_pos}")
```

### Visual Simulation

```python
# Run with real-time visualization
result = sim.run(headless=False)
```

## Core Classes

### SimulationConfig

Configuration dataclass containing all simulation parameters:

- **Area and Timing**: `area_w`, `area_h`, `dt`, `duration`
- **Dynamics**: `speed`, `comm_r`, `n_agents`, launch parameters
- **Pheromone System**: `phi_init`, `phi_max`, `phi_ant`, `phi_conn`, `phi_internal`, `phi_decay`, `mu`
- **Lattice**: `lattice_angle_deg`, `step_dist_factor`
- **User Placement**: `user_mode`, `user_fixed`, `user_margin`
- **Communication**: `data_interval` for data flooding
- **Control**: `seed`, `terminate_on_success`

### SimulationResult

Results dataclass containing simulation outcomes:

- `success`: Whether user was found within communication range
- `success_time`: Time when success was achieved (seconds)
- `final_time`: Total simulation time
- `user_pos`: Final user position
- `base_pos`: Base station position
- `num_nodes`: Number of active nodes at end
- `num_agents_launched`: Number of agents launched
- `nodes_created`: Total nodes created during simulation
- `min_agent_user_distance`: Minimum distance between any agent and user
- `last_agent_user_distance`: Final distance between closest agent and user
- `success_via_min_hop_at_time`: Whether success occurred via minimal hop path
- `optimal_hop_len_at_end`: Optimal hop count to user at simulation end
- `min_hop_len_at_end`: Minimum hop count to user at simulation end

### SmavNet2D

Main simulation class providing the SMAVNET implementation:

#### Key Methods

- `__init__(config)`: Initialize simulation with configuration
- `reset(seed=None)`: Reset simulation state for new run
- `run(headless=True, max_time=None)`: Run complete simulation
- `step()`: Advance simulation by one timestep
- `get_state_snapshot()`: Get lightweight state metrics

## Algorithm Details

### Pheromone System

The system uses three types of pheromone reinforcement:

1. **Ant Pheromone** (`phi_ant`): Deposited by agents at nodes they reference
2. **Internal Pheromone** (`phi_internal`): Reinforces nodes that have forward neighbors
3. **Connection Pheromone** (`phi_conn`): Reinforces nodes on least-hop paths to user

### Branch Selection

Agents use Deneubourg-style probabilistic branch selection:

- Probability based on pheromone levels at left and right branches
- Incorporates distance-based bias (`mu` parameter)
- Uses sigmoid-like probability function for decision making

### Data Flooding

The system implements a two-phase communication protocol:

1. **Data Flooding**: Messages propagate from base through the network
2. **Control Propagation**: Global hop counts are advertised back through the network
3. **Least-Hop Identification**: Nodes determine if they're on optimal paths

### Retraction Behavior

When node pheromone decays to zero:

- Node disappears and agent becomes retracting
- Agent follows maximum pheromone path back to base
- Spiral search for reconnection if path is lost
- Continuous relaunch system keeps agents active

## Experiments

The project includes comprehensive experimental tools in `experiments_scalability.py`:

### Scalability Analysis

```python
from experiments_scalability import run_scalability_experiment

# Test success probability vs swarm size
run_scalability_experiment(
    swarm_sizes=list(range(5, 21)),
    num_trials=500,
    duration_s=1800.0,
    area_w=800.0,
    area_h=600.0
)
```

### Success Probability vs Time

```python
from experiments_scalability import run_success_prob_vs_time_experiment

# Analyze cumulative success probability over time
run_success_prob_vs_time_experiment(
    n_agents=15,
    num_trials=10000,
    max_duration_s=1800.0
)
```

### User Distribution Analysis

```python
from experiments_scalability import plot_user_distribution_with_coverage

# Visualize user placement and search coverage
plot_user_distribution_with_coverage(
    n_agents=15,
    num_trials=500,
    duration_s=1800.0
)
```

## Configuration Examples

### High-Speed Search

```python
cfg = SimulationConfig(
    n_agents=25,
    speed=15.0,
    comm_r=120.0,
    duration=600.0,
    terminate_on_success=True
)
```

### Large Area Coverage

```python
cfg = SimulationConfig(
    n_agents=30,
    area_w=1200.0,
    area_h=1000.0,
    comm_r=100.0,
    duration=1800.0
)
```

### Deterministic Testing

```python
cfg = SimulationConfig(
    n_agents=18,
    seed=12345,
    user_mode="fixed",
    user_fixed=(400.0, 300.0)
)
```

## Performance Considerations

- **Headless Mode**: Use `headless=True` for maximum performance in experiments
- **Termination**: Set `terminate_on_success=True` to stop simulation upon first success

## File Structure

```
├── smavnet_sim_final.py          # Main simulation implementation
├── experiments_scalability.py    # Experimental analysis tools
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── Ant-based swarming with positionless micro air vehicles for communication relay.pdf
```

## Research Context

This implementation is based on the research paper:

**Hauert, Sabine & Winkler, Laurent & Zufferey, Jean-Christophe & Floreano, Dario. (2008). Ant-based Swarming with Positionless Micro Air Vehicles for Communication Relay. Swarm Intelligence. 2. 10.1007/s11721-008-0013-5.**

The original work presents a 3D bio-inspired communication relay system using Micro Air Vehicles (MAVs) that employ ant colony optimization principles. This project adapts the 3D algorithm for 2D simulation while maintaining the core biological inspiration and swarm intelligence concepts.

### Key Contributions from Original Paper

- Bio-inspired swarm intelligence for communication networks
- Self-organizing network formation and maintenance using pheromone trails
- Adaptive search strategies for user location
- Robust communication relay capabilities in dynamic environments
- Positionless navigation using relative positioning and communication

### Our 2D Adaptation

This implementation demonstrates:

- **2D Lattice Structure**: Hexagonal-like lattice based on (i,j) coordinates
- **Pheromone-based Path Selection**: Deneubourg-style probabilistic branch selection
- **Data Flooding Protocol**: Two-phase communication for least-hop route identification
- **Dynamic Retraction**: Agents follow maximum pheromone paths back to base
- **Continuous Launch System**: Maintains active agent population through relaunching
- **Comprehensive Experiments**: Scalability analysis and performance evaluation tools

## License

This project is developed for academic research purposes as part of ME5253 Network Dynamics course work at IIT Madras.

### Academic Use

This implementation is provided for educational and research purposes. The original SMAVNET algorithm is based on the research paper by Hauert et al. (2008). Users are encouraged to:

- Cite the original paper when using this implementation in academic work
- Follow academic integrity guidelines when building upon this work
- Respect the intellectual property of the original authors

### Original Paper Citation

When referencing this implementation, please also cite the original work:

```
Hauert, S., Winkler, L., Zufferey, J. C., & Floreano, D. (2008). 
Ant-based swarming with positionless micro air vehicles for communication relay. 
Swarm Intelligence, 2(1), 73-95. https://doi.org/10.1007/s11721-008-0013-5
```

### Disclaimer

This implementation is an independent adaptation of the original SMAVNET algorithm for 2D simulation. The authors of this implementation are not affiliated with the original paper authors and do not claim ownership of the underlying SMAVNET concept or algorithm.

## Contributing

This is an academic project. For questions or suggestions, please contact the authors:

- **Mayank Chandak** (ME22B224): me22b224@smail.iitm.ac.in
- **Aldis Daniel** (ME22B070): me22b070@smail.iitm.ac.in