# SMAVNET 2D Simulator

A Python implementation of the Swarm-based Micro Air Vehicle Network (SMAVNET) system for communication relay in 2D environments. This project includes both a baseline reproduction of the original algorithm and an extended version with wind disturbances and drift mitigation mechanisms.

## Authors

- **Mayank Chandak** (ME22B224)
- **Aldis Daniel** (ME22B070)

## Overview

SMAVNET is a bio-inspired communication relay system that uses Micro Air Vehicles (MAVs) to establish and maintain communication networks. The system employs ant colony optimization principles where MAVs act as mobile communication nodes, creating a dynamic network that can adapt to changing conditions and user locations.

This repository contains two implementations:

1. **`smavnet_sim_final.py`**: Baseline implementation that faithfully reproduces the original SMAVNET algorithm from the research paper, adapted for 2D simulation. This serves as a reference implementation demonstrating the core ant-based swarming behavior without environmental disturbances.

2. **`smavnet_sim_final_wind.py`**: Extended implementation developed for the course project, adding:
   - **Wind Disturbance Model**: Realistic time-varying wind field affecting agent motion
   - **Node Replacement Mechanism**: Probabilistic replacement policy to mitigate drift accumulation
   - **Drift Metrics**: Comprehensive tracking of geometric stability and drift reduction
   - **Enhanced Visualization**: Improved renderer with wind visualization and detailed metrics

### Key Features

- **Bio-inspired Algorithm**: Implements pheromone-based path selection similar to ant colony optimization
- **Dynamic Network Formation**: MAVs autonomously create and maintain communication relay networks
- **User Search Capability**: System can locate and establish communication with users in the search area
- **Wind Robustness** (extended version): Handles environmental disturbances with drift mitigation
- **Configurable Parameters**: Extensive configuration options for simulation parameters
- **Headless and Visual Modes**: Fast headless simulation for experiments and optional real-time visualization

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup Steps

1. **Clone or download this repository**

2. **Navigate to the project directory**:
   ```bash
   cd Network_dynamics_course_project
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `numpy` - For numerical computations and array operations
   - `matplotlib` - For visualization and plotting
   - `tqdm` - For progress bars during long experiments

4. **Verify installation**:
   ```bash
   python -c "import numpy, matplotlib, tqdm; print('All dependencies installed successfully!')"
   ```

### Quick Test

Run a quick test to verify everything works:

```bash
python smavnet_sim_final.py
```

This should run a short simulation and print results. For the wind implementation:

```bash
python smavnet_sim_final_wind.py
```

## Quick Start

### Basic Usage (Baseline Implementation)

The baseline implementation (`smavnet_sim_final.py`) reproduces the original SMAVNET algorithm:

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

### Extended Usage (Wind Implementation)

The extended implementation (`smavnet_sim_final_wind.py`) adds wind disturbances and replacement mechanisms:

```python
from smavnet_sim_final_wind import SimulationConfig, SmavNet2D

# Create simulation configuration with wind
cfg = SimulationConfig(
    n_agents=18,
    comm_r=100.0,
    duration=1800.0,
    enable_wind=True,              # Enable wind disturbances
    wind_speed=2.0,                # Wind speed (m/s)
    enable_replacement=True,       # Enable node replacement
    replacement_min_age=30.0,      # Minimum node age before replacement
    replacement_time_scale=120.0,  # Replacement probability time scale
    seed=42
)

# Create and run simulation
sim = SmavNet2D(cfg)
result = sim.run(headless=True)

# Check results including drift metrics
print(f"Success: {result.success}")
print(f"Success Time: {result.success_time}s")
print(f"Average Drift: {result.avg_drift:.4f} (normalized)")
print(f"Average Drift: {result.avg_drift_raw:.2f}m (raw)")
if result.num_swaps > 0:
    print(f"Number of Swaps: {result.num_swaps}")
    print(f"Average Drift Reduction: {result.avg_drift_reduction:.4f} per swap")
```

### Visual Simulation

Both implementations support real-time visualization:

```python
# Run with real-time visualization
result = sim.run(headless=False)
```

The visual mode shows:
- Agent positions and states (ANT, NODE, CIRCLING, RETRACTING, HOMING)
- Node positions with pheromone levels
- Communication links between nodes
- Wind vector (if enabled)
- Agent-to-destination connections
- Success status and metrics

## Implementation Files

### `smavnet_sim_final.py` - Baseline Implementation

This file contains the baseline SMAVNET 2D implementation that faithfully reproduces the original algorithm from the research paper. It serves as a reference implementation demonstrating:

- Core ant-based swarming behavior
- Pheromone-based path selection
- Data flooding and control propagation
- Dynamic node creation and retraction
- Original algorithm behavior without environmental disturbances

**Use this file when:**
- Reproducing baseline results from the original paper
- Understanding the core SMAVNET algorithm
- Comparing against the original implementation
- Running experiments without wind disturbances

### `smavnet_sim_final_wind.py` - Extended Implementation

Extended implementation with wind disturbances and drift mitigation mechanisms. Includes all baseline features plus wind model, replacement policy, drift metrics, and enhanced visualization. See [Wind Disturbance Model](#wind-disturbance-model) and [Node Replacement Mechanism](#node-replacement-mechanism) for details.

**Use this file when:**
- Studying robustness to environmental disturbances
- Analyzing drift mitigation strategies
- Running experiments with wind conditions
- Evaluating replacement mechanism effectiveness

### `experiments_wind.py` - Wind Experiments

Experimental tools for analyzing the wind implementation, including replacement comparison, drift analysis, and scalability evaluation. See the [Experiments](#experiments) section for detailed usage.

## Core Classes

### SimulationConfig

Configuration dataclass containing all simulation parameters. Both implementations share common parameters:

**Common Parameters:**
- **Area and Timing**: `area_w`, `area_h`, `dt`, `duration`
- **Dynamics**: `speed`, `comm_r`, `n_agents`, launch parameters
- **Pheromone System**: `phi_init`, `phi_max`, `phi_ant`, `phi_conn`, `phi_internal`, `phi_decay`, `mu`
- **Lattice**: `lattice_angle_deg`, `step_dist_factor`
- **User Placement**: `user_mode`, `user_fixed`, `user_margin`
- **Communication**: `data_interval` for data flooding
- **Control**: `seed`, `terminate_on_success`

**Extended Parameters** (wind implementation only):
- **Wind Disturbances**: `enable_wind`, `wind_speed`, `wind_direction_mean`, `wind_direction_std`, `wind_angular_frequency`, `wind_magnitude_variation`, `wind_effect_scale`
- **Arrival Detection**: `arrival_distance_factor`, `arrival_threshold`, `arrival_time_factor`
- **Replacement Policy**: `enable_replacement`, `replacement_min_age`, `replacement_time_scale`
- **Logging**: `log` (enable/disable debug prints)

### SimulationResult

Results dataclass containing simulation outcomes. The baseline version includes:

**Baseline Results:**
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

**Extended Results** (wind implementation only):
- `num_swaps`: Total number of node replacement events
- `swap_times`: List of times when replacements occurred
- `swap_nodes`: List of node coordinates where replacements occurred
- `swap_ages`: List of node ages at replacement time
- `avg_drift`: Time-averaged drift (normalized by comm_r)
- `avg_drift_raw`: Time-averaged drift in meters
- `swap_pre_drifts`: Drift values before each replacement
- `swap_post_drifts`: Drift values after each replacement
- `swap_drift_reductions`: Drift reduction for each replacement event
- `avg_drift_reduction`: Average drift reduction per swap (normalized)
- `avg_drift_reduction_raw`: Average drift reduction per swap in meters

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

## Wind Disturbance Model

The extended implementation includes a realistic wind model that affects agent motion:

### Wind Components

The wind vector combines three components:
1. **Deterministic Sinusoidal Variation**: Low-frequency periodic changes in wind direction
2. **Random Walk Component**: Stochastic gusts and turbulence
3. **Magnitude Variation**: Random fluctuations in wind speed

### Wind Parameters

- `wind_speed`: Nominal wind speed magnitude (m/s)
- `wind_direction_mean`: Mean wind direction in degrees (0=north, 90=east)
- `wind_direction_std`: Standard deviation of wind direction variation
- `wind_angular_frequency`: Frequency of sinusoidal variation (rad/s)
- `wind_magnitude_variation`: Variation coefficient for wind speed
- `wind_effect_scale`: Scaling factor to adjust wind influence on motion

### Wind Effect on Agents

Wind affects agent motion in two ways:
1. **Direct Displacement**: Wind vector is added to agent velocity each timestep
2. **Orbit Center Drift**: For circling agents, the orbit center itself drifts with wind, maintaining realistic physics

## Node Replacement Mechanism

To mitigate drift accumulation under wind, the extended implementation includes a probabilistic node replacement policy:

### Replacement Logic

When an agent arrives at an existing node:
1. **Age Calculation**: Node age = current_time - node_creation_time
2. **Probability Evaluation**: Replacement probability increases exponentially with age:
   ```
   p_replace = 1 - exp(-(age - min_age) / tau)
   ```
3. **Replacement Decision**: If probability threshold is met, new agent takes over the node
4. **Drift Reduction**: Previous holder transitions to HOMING state, moving toward new node's orbit center

### Replacement Parameters

- `enable_replacement`: Enable/disable replacement mechanism
- `replacement_min_age`: Minimum node age (seconds) before replacement can occur
- `replacement_time_scale`: Time scale parameter (tau) controlling probability growth

### Replacement Benefits

- **Drift Mitigation**: Fresh agents reduce accumulated drift at nodes
- **Network Stability**: Maintains geometric coherence under persistent disturbances
- **Self-Organizing**: No centralized control required
- **Scalable**: Replacement rate scales naturally with swarm size

## Experiments

The project includes two experiment files for baseline and wind implementations:

### Baseline Experiments (`experiments.py`)

Run experiments for the baseline implementation:

```bash
python experiments.py
```

This generates plots for:
- Scalability analysis (success probability vs swarm size)
- Success probability vs time
- User distribution and coverage

### Wind Experiments (`experiments_wind.py`)

Run comprehensive experiments for the wind implementation:

```bash
python experiments_wind.py
```

This will:
1. Create a timestamped output directory (e.g., `replacement_drift_experiments_20240101_120000/`)
2. Run replacement comparison experiment (with vs without replacement)
3. Run drift analysis experiment
4. Save all plots to the output directory

**Output plots include:**
- `replacement_comparison_drift.png`: Average drift comparison
- `replacement_swaps_vs_swarm_size.png`: Number of swaps vs swarm size
- `replacement_pre_post_drift_n15_trials500.png`: Pre vs post drift scatter
- `replacement_reduction_vs_pre_drift_n15_trials500.png`: Reduction effectiveness
- `replacement_reduction_over_time_n15_trials500.png`: Reduction over time
- `replacement_reduction_vs_age_n15_trials500.png`: Reduction vs node age

**Programmatic Usage:**

```python
from experiments_wind import run_replacement_comparison_experiment, run_drift_vs_replacement_experiment

# Compare drift with and without replacement
run_replacement_comparison_experiment(
    swarm_sizes=list(range(5, 21)),
    num_trials=500,
    duration_s=1800.0,
    area_w=800.0,
    area_h=600.0,
    wind_speed=2.0,
    output_dir="./results"
)

# Analyze replacement effectiveness
run_drift_vs_replacement_experiment(
    n_agents=15,
    num_trials=500,
    duration_s=1800.0,
    wind_speed=2.0,
    output_dir="./results"
)
```

## Configuration Examples

### Baseline Configuration

```python
from smavnet_sim_final import SimulationConfig, SmavNet2D

cfg = SimulationConfig(
    n_agents=18,
    comm_r=100.0,
    duration=900.0,
    speed=10.0,
    seed=42,
    user_mode="uniform"
)
```

### Wind Configuration (Low Wind)

```python
from smavnet_sim_final_wind import SimulationConfig, SmavNet2D

cfg = SimulationConfig(
    n_agents=18,
    comm_r=100.0,
    duration=1800.0,
    enable_wind=True,
    wind_speed=0.5,              # Light wind
    enable_replacement=True,
    replacement_min_age=30.0,
    replacement_time_scale=120.0,
    seed=42
)
```

### Wind Configuration (Moderate Wind)

```python
from smavnet_sim_final_wind import SimulationConfig, SmavNet2D

cfg = SimulationConfig(
    n_agents=18,
    comm_r=100.0,
    duration=1800.0,
    enable_wind=True,
    wind_speed=2.0,              # Moderate wind
    wind_direction_mean=45.0,    # Northeast
    wind_direction_std=30.0,     # Variable direction
    enable_replacement=True,
    replacement_min_age=30.0,
    replacement_time_scale=120.0,
    seed=42
)
```

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
- **Batch Experiments**: Wind experiments can take significant time (500 trials × multiple swarm sizes). Consider running overnight or on a compute cluster for large-scale analysis
- **Memory**: Long simulations with many agents may require increased memory allocation

## File Structure

```
Network_dynamics_course_project/
├── smavnet_sim_final.py          # Baseline implementation
├── smavnet_sim_final_wind.py     # Extended implementation (wind + replacement)
├── experiments.py                # Baseline experiments
├── experiments_wind.py           # Wind experiments
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── pics/                         # Generated plots and figures
└── Ant-based swarming with positionless micro air vehicles for communication relay.pdf
```

See the [Implementation Files](#implementation-files) section for detailed descriptions of each file.

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

The baseline implementation (`smavnet_sim_final.py`) demonstrates:

- **2D Lattice Structure**: Hexagonal-like lattice based on (i,j) coordinates
- **Pheromone-based Path Selection**: Deneubourg-style probabilistic branch selection
- **Data Flooding Protocol**: Two-phase communication for least-hop route identification
- **Dynamic Retraction**: Agents follow maximum pheromone paths back to base
- **Continuous Launch System**: Maintains active agent population through relaunching
- **Comprehensive Experiments**: Scalability analysis and performance evaluation tools

### Course Project Extensions

The extended implementation (`smavnet_sim_final_wind.py`) adds:

- **Wind Disturbance Model**: Realistic time-varying wind field affecting agent motion
  - Deterministic sinusoidal variation for low-frequency drift
  - Random walk component for stochastic gusts
  - Magnitude variation for speed fluctuations
  
- **Node Replacement Mechanism**: Probabilistic replacement policy to mitigate drift
  - Age-based replacement probability: `p = 1 - exp(-(age - min_age) / tau)`
  - Automatic drift reduction through fresh agent assignment
  - Self-organizing and scalable design
  
- **Drift Metrics**: Comprehensive geometric stability tracking
  - Time-averaged drift across all nodes
  - Per-swap drift reduction measurements
  - Normalized and raw drift values for analysis
  
- **Enhanced Visualization**: Improved renderer with wind arrows, pheromone annotations, and detailed metrics

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

- **Mayank Chandak** (ME22B224): [me22b224@smail.iitm.ac.in](mailto:me22b224@smail.iitm.ac.in)
- **Aldis Daniel** (ME22B070): [me22b070@smail.iitm.ac.in](mailto:me22b070@smail.iitm.ac.in)