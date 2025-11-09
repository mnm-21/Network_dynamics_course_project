import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np


# --------------------------
# Public API dataclasses
# --------------------------


@dataclass
class SimulationConfig:
    """Configuration parameters for SMAVNET 2D simulation."""
    # Area and timing
    area_w: float = 800.0
    area_h: float = 600.0
    dt: float = 0.1
    duration: float = 900.0

    # Dynamics
    speed: float = 10.0
    comm_r: float = 100.0
    n_agents: int = 18
    launch_mean: float = 15.0
    launch_jitter: float = 7.5
    launch_duration: float = 4.0

    # Pheromone
    phi_init: float = 0.7
    phi_max: float = 1.0
    phi_ant: float = 0.0001
    phi_conn: float = 0.01
    phi_internal: float = 0.001
    phi_decay: float = 0.001
    mu: float = 0.75

    # Lattice
    lattice_angle_deg: float = 30.0
    step_dist_factor: float = 0.95

    # User placement
    user_mode: str = "uniform"
    user_fixed: Optional[Tuple[float, float]] = None
    user_margin: float = 50.0
    user_gaussian_center: Optional[Tuple[float, float]] = None
    user_gaussian_sigma: float = 120.0

    # Communication
    data_interval: float = 6.0

    # Randomness
    seed: Optional[int] = None

    # Termination
    terminate_on_success: bool = False

    # Wind disturbances
    enable_wind: bool = False
    wind_speed: float = 2.0
    wind_direction_mean: float = 45.0
    wind_direction_std: float = 0.0
    wind_angular_frequency: float = 0.1
    wind_magnitude_variation: float = 0.5
    wind_effect_scale: float = 0.5
    arrival_distance_factor: float = 1.0

    # Arrival detection
    arrival_threshold: float = 12.0
    arrival_time_factor: float = 1.05

    # Replacement policy
    enable_replacement: bool = True
    replacement_min_age: float = 30.0
    replacement_time_scale: float = 120.0
    
    # Logging
    log: bool = False


@dataclass
class SimulationResult:
    """Results from a SMAVNET simulation run."""
    success: bool
    success_time: Optional[float]
    final_time: float
    user_pos: Tuple[float, float]
    base_pos: Tuple[float, float]
    num_nodes: int
    num_agents_launched: int
    nodes_created: int
    min_agent_user_distance: float
    last_agent_user_distance: float
    success_via_min_hop_at_time: Optional[bool]
    optimal_hop_len_at_end: Optional[int]
    min_hop_len_at_end: Optional[int]
    # Swap/replacement metrics
    num_swaps: int = 0
    swap_times: List[float] = field(default_factory=list)
    swap_nodes: List[Tuple[int, int]] = field(default_factory=list)
    swap_ages: List[float] = field(default_factory=list)
    # Drift metrics
    avg_drift: float = 0.0  # time-averaged drift across all nodes, normalized by comm_r
    avg_drift_raw: float = 0.0  # time-averaged drift in meters (not normalized)
    # Per-swap drift metrics
    swap_pre_drifts: List[float] = field(default_factory=list)  # drift before each swap
    swap_post_drifts: List[float] = field(default_factory=list)  # drift after each swap
    swap_drift_reductions: List[float] = field(default_factory=list)  # reduction (pre - post) for each swap
    avg_drift_reduction: float = 0.0  # average drift reduction per swap, normalized by comm_r
    avg_drift_reduction_raw: float = 0.0  # average drift reduction per swap in meters


# --------------------------
# Internal helper types
# --------------------------


class Node:
    def __init__(self, i: int, j: int, pos: np.ndarray, phi_init: float):
        self.i = i
        self.j = j
        self.pos = pos.copy()
        self.phi = phi_init
        self.exists = True
        self.hop_to_base = float("inf")
        self.hop_to_user = float("inf")
        self.neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
        self.agent_id: Optional[int] = None
        self.time_created: float = 0.0

        # Communication bookkeeping
        self.seen_data_ids: Dict[str, int] = {}
        self.known_global_hopcounts: Dict[str, int] = {}
        self.on_least_hop: bool = False


class Agent:
    def __init__(self, agent_id: int, base_pos: np.ndarray, speed: float, launch_duration: float):
        self.id = agent_id
        self.pos = base_pos.copy() + np.random.normal(scale=1.0, size=2)
        self.heading = random.uniform(0.0, 2.0 * math.pi)
        self.v = speed
        self.state = "ON_GROUND"
        self.role = "ANT"
        self.launch_start_time: Optional[float] = None
        self.launch_duration = launch_duration
        self.ref_ij = (0, 0)
        self.dest_ij: Optional[Tuple[int, int]] = None
        self.branch_sign: Optional[int] = None
        self.orbit_radius = 10.0
        self.orbit_ang = 0.0
        self.angular_speed = 0.8
        self.retract_target: Optional[Tuple[int, int]] = None
        self.spiral_theta = 0.0
        self.spiral_r = 0.0
        self.spiral_omega = 2.0
        self.held_node: Optional[Tuple[int, int]] = None
        self.travel_start_time: Optional[float] = None
        self.expected_flight_time: Optional[float] = None
        self.orbit_center: Optional[np.ndarray] = None
        self.homing_target: Optional[np.ndarray] = None

    def begin_launch(self, current_time: float, base_pos: np.ndarray):
        if self.state == "ON_GROUND":
            self.state = "LAUNCHING"
            self.launch_start_time = current_time
            self.pos = base_pos.copy() + np.random.normal(scale=0.5, size=2)
            self.ref_ij = (0, 0)
            self.dest_ij = None
            self.branch_sign = None


class SmavNet2D:
    """SMAVNET 2D simulator with configurable parameters and optional visualization."""
    
    def __init__(self, config: SimulationConfig):
        """Initialize the SMAVNET simulation with given configuration."""
        self.cfg = config
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # Geometry
        self.base_pos = np.array([self.cfg.area_w / 2.0, 0.0])
        self.user_pos = self._sample_user_pos()
        self.dt = self.cfg.dt

        base_dir = np.array([0.0, 1.0])
        self.v_left = self._rotate(base_dir, +self.cfg.lattice_angle_deg)
        self.v_right = self._rotate(base_dir, -self.cfg.lattice_angle_deg)
        self.step_dist = self.cfg.comm_r * self.cfg.step_dist_factor

        self.t = 0.0
        self.step_idx = 0
        self.node_table: Dict[Tuple[int, int], Node] = {}
        self.agents: List[Agent] = [
            Agent(i, self.base_pos, self.cfg.speed, self.cfg.launch_duration)
            for i in range(self.cfg.n_agents)
        ]
        self._init_nodes()
        self.next_launch_time = self._sample_launch_interval()

        self.next_data_time = 0.0
        self.active_data_msgs: Dict[str, Dict] = {}
        self.active_control_msgs: Dict[str, Dict] = {}
        self.user_first_hop: Dict[str, int] = {}
        self.user_received: Set[str] = set()

        self.success = False
        self.success_time: Optional[float] = None
        self.min_agent_user_distance = float("inf")
        self.last_agent_user_distance = float("inf")
        self.nodes_created = 1
        self._success_via_min_hop_flag: Optional[bool] = None
        
        self.num_swaps = 0
        self.swap_times: List[float] = []
        self.swap_nodes: List[Tuple[int, int]] = []
        self.swap_ages: List[float] = []
        self.swap_pre_drifts: List[float] = []
        self.swap_post_drifts: List[float] = []
        self.swap_drift_reductions: List[float] = []
        
        self.drift_sum: float = 0.0
        self.drift_sample_count: int = 0

        self.wind_vector = np.array([0.0, 0.0])
        self.wind_angle = 0.0
        if self.cfg.enable_wind:
            self._update_wind()

    # ---------------------- utils ----------------------
    @staticmethod
    def _rotate(vec: np.ndarray, degrees: float) -> np.ndarray:
        """Rotate vector by given degrees counterclockwise."""
        th = math.radians(degrees)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        return R.dot(vec)

    def _node_pos_from_ij(self, i: int, j: int) -> np.ndarray:
        """Convert lattice coordinates (i,j) to world position."""
        return self.base_pos + self.step_dist * (i * self.v_left + j * self.v_right)

    def _connected(self, a_pos: np.ndarray, b_pos: np.ndarray) -> bool:
        """Check if two positions are within communication range."""
        return np.linalg.norm(a_pos - b_pos) <= self.cfg.comm_r + 1e-6

    def _init_nodes(self) -> None:
        """Initialize base node at origin."""
        base_node = Node(0, 0, self.base_pos, self.cfg.phi_max)
        base_node.hop_to_base = 0
        self.node_table[(0, 0)] = base_node

    def _sample_launch_interval(self) -> float:
        """Sample next agent launch interval."""
        return max(0.0, random.gauss(self.cfg.launch_mean, self.cfg.launch_jitter))


    def _sample_user_pos(self) -> np.ndarray:
        """Sample user position within search region based on user_mode."""
        if self.cfg.user_mode == "fixed" and self.cfg.user_fixed is not None:
            return np.array(self.cfg.user_fixed, dtype=float)

        base_x, base_y = self.base_pos
        comm_r = self.cfg.comm_r
        angle_deg = self.cfg.lattice_angle_deg  
        angle_rad = math.radians(angle_deg)

        left_dir = self._rotate(np.array([0.0, 1.0]), -angle_deg)
        right_dir = self._rotate(np.array([0.0, 1.0]), +angle_deg)

        # Compute perpendicular directions to define search region boundaries
        # These define the left and right edges of the searchable area
        left_perp = self._rotate(left_dir, -90.0)
        right_perp = self._rotate(right_dir, +90.0)

        # Offset lines: parallel to lattice directions, offset by comm_r from base
        # These define the searchable region (between left and right boundaries)
        left_origin = np.array([base_x, base_y]) + comm_r * left_perp
        right_origin = np.array([base_x, base_y]) + comm_r * right_perp

        # Y-range: ensure user is beyond initial communication range
        min_y = comm_r * math.cos(angle_rad) * 2
        max_y = self.cfg.area_h - self.cfg.user_margin

        y = random.uniform(min_y, max_y)

        # Find x-coordinates on left and right boundaries at this y
        # Solve: y = origin_y + t*dy for parameter t, then x = origin_x + t*dx
        t_left = (y - left_origin[1]) / left_dir[1]
        t_right = (y - right_origin[1]) / right_dir[1]

        x_left = left_origin[0] + t_left * left_dir[0]
        x_right = right_origin[0] + t_right * right_dir[0]
        # Sample uniformly within the searchable region
        x = random.uniform(x_left, x_right)

        return np.array([x, y])

    def _update_wind(self) -> None:
        """Update wind vector with time-varying disturbances.
        
        Wind model combines:
        - Deterministic sinusoidal variation (low-frequency drift)
        - Random walk component (stochastic gusts)
        - Magnitude variation (speed fluctuations)
        """
        base_angle = math.radians(self.cfg.wind_direction_mean)
        # Sinusoidal component provides smooth periodic variation
        sinusoidal_component = math.sin(self.cfg.wind_angular_frequency * self.t)
        # Random walk accumulates small random changes for realistic turbulence
        angle_change = np.random.normal(0.0, math.radians(self.cfg.wind_direction_std) / 10.0)
        self.wind_angle += angle_change
        # Combine base direction with accumulated random walk and sinusoidal variation
        angle_variation = sinusoidal_component * math.radians(self.cfg.wind_direction_std)
        current_angle = base_angle + self.wind_angle + angle_variation
        # Vary wind speed magnitude with random fluctuations
        magnitude_variation = 1.0 + self.cfg.wind_magnitude_variation * np.random.normal(0.0, 0.3)
        current_magnitude = self.cfg.wind_speed * magnitude_variation
        # Convert to Cartesian velocity vector
        self.wind_vector = np.array([
            current_magnitude * math.cos(current_angle),
            current_magnitude * math.sin(current_angle)
        ])
    
    def _get_wind_influence(self) -> np.ndarray:
        """Get current wind velocity vector."""
        if self.cfg.enable_wind:
            return self.wind_vector.copy() * float(self.cfg.wind_effect_scale)
        return np.array([0.0, 0.0])

    # ---------------------- public API ----------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset simulation state for a new run with optional new seed."""
        if seed is not None:
            self.cfg.seed = seed
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
        self.user_pos = self._sample_user_pos()
        base_dir = np.array([0.0, 1.0])
        self.v_left = self._rotate(base_dir, +self.cfg.lattice_angle_deg)
        self.v_right = self._rotate(base_dir, -self.cfg.lattice_angle_deg)
        self.step_dist = self.cfg.comm_r * self.cfg.step_dist_factor
        self.t = 0.0
        self.step_idx = 0
        self.node_table.clear()
        self._init_nodes()
        self.agents = [Agent(i, self.base_pos, self.cfg.speed, self.cfg.launch_duration) for i in range(self.cfg.n_agents)]
        self.next_launch_time = self._sample_launch_interval()
        self.next_data_time = 0.0
        self.active_data_msgs.clear()
        self.active_control_msgs.clear()
        self.user_first_hop.clear()
        self.user_received.clear()
        self.success = False
        self.success_time = None
        self.min_agent_user_distance = float("inf")
        self.last_agent_user_distance = float("inf")
        self.nodes_created = 1
        self._success_via_min_hop_flag = None
        self.num_swaps = 0
        self.swap_times.clear()
        self.swap_nodes.clear()
        self.swap_ages.clear()
        self.swap_pre_drifts.clear()
        self.swap_post_drifts.clear()
        self.swap_drift_reductions.clear()
        self.drift_sum = 0.0
        self.drift_sample_count = 0
        self.wind_vector = np.array([0.0, 0.0])
        self.wind_angle = 0.0
        if self.cfg.enable_wind:
            self._update_wind()

    def run(self, headless: bool = True, max_time: Optional[float] = None) -> SimulationResult:
        """Run the simulation and return results."""
        end_time = min(self.cfg.duration, max_time) if max_time is not None else self.cfg.duration
        renderer = None
        if not headless:
            renderer = _Renderer(self)
            renderer.setup()
        
        while self.t < end_time:
            self.step()
            if self.cfg.terminate_on_success and self.success:
                break
            if renderer is not None and renderer.should_draw(self.t):
                renderer.draw(self.t)
        min_hop_end = self._min_hop_nodes_to_user()
        
        avg_drift_raw = 0.0
        avg_drift = 0.0
        if self.drift_sample_count > 0:
            avg_drift_raw = self.drift_sum / float(self.drift_sample_count)
            avg_drift = avg_drift_raw / self.cfg.comm_r
        
        avg_drift_reduction_raw = 0.0
        avg_drift_reduction = 0.0
        if len(self.swap_drift_reductions) > 0:
            avg_drift_reduction_raw = sum(self.swap_drift_reductions) / float(len(self.swap_drift_reductions))
            avg_drift_reduction = avg_drift_reduction_raw / self.cfg.comm_r
        
        return SimulationResult(
            success=self.success,
            success_time=self.success_time,
            final_time=self.t,
            user_pos=(float(self.user_pos[0]), float(self.user_pos[1])),
            base_pos=(float(self.base_pos[0]), float(self.base_pos[1])),
            num_nodes=sum(1 for n in self.node_table.values() if n.exists),
            num_agents_launched=sum(1 for a in self.agents if a.state != "ON_GROUND"),
            nodes_created=self.nodes_created,
            min_agent_user_distance=self.min_agent_user_distance,
            last_agent_user_distance=self.last_agent_user_distance,
            success_via_min_hop_at_time=self._success_via_min_hop_flag,
            optimal_hop_len_at_end=min_hop_end,
            min_hop_len_at_end=min_hop_end,
            num_swaps=self.num_swaps,
            swap_times=self.swap_times.copy(),
            swap_nodes=self.swap_nodes.copy(),
            swap_ages=self.swap_ages.copy(),
            avg_drift=avg_drift,
            avg_drift_raw=avg_drift_raw,
            swap_pre_drifts=self.swap_pre_drifts.copy(),
            swap_post_drifts=self.swap_post_drifts.copy(),
            swap_drift_reductions=self.swap_drift_reductions.copy(),
            avg_drift_reduction=avg_drift_reduction,
            avg_drift_reduction_raw=avg_drift_reduction_raw,
        )

    def get_state_snapshot(self) -> Dict[str, float]:
        """Get lightweight snapshot of current simulation state."""
        return {
            "t": self.t,
            "num_nodes": float(sum(1 for n in self.node_table.values() if n.exists)),
            "num_agents_launched": float(sum(1 for a in self.agents if a.state != "ON_GROUND")),
            "success": float(1.0 if self.success else 0.0),
        }

    # ---------------------- core step logic ----------------------
    def step(self) -> None:
        """Advance simulation by one timestep.
        
        Execution order:
        1. Update wind (if enabled)
        2. Launch agents from base
        3. Move agents
        4. Detect arrivals and create/replace nodes
        5. Recompute routing (hop counts)
        6. Propagate data and control messages
        7. Update pheromone levels
        8. Handle node decay and agent retraction
        9. Check for success condition
        10. Compute drift metrics
        """
        t = self.t
        dt = self.dt

        if self.cfg.enable_wind:
            self._update_wind()

        if t >= self.next_launch_time:
            launched = False
            for ag in self.agents:
                if ag.state == "ON_GROUND":
                    if float(np.linalg.norm(ag.pos - self.base_pos)) <= self.cfg.comm_r:
                        ag.begin_launch(t, self.base_pos)
                        launched = True
                        break
            self.next_launch_time = t + self._sample_launch_interval()

        for ag in self.agents:
            self._agent_step_motion(ag, dt, t)

        for ag in self.agents:
            if ag.state == "ACTIVE" and ag.dest_ij is not None:
                dest = ag.dest_ij
                dest_pos = self._node_pos_from_ij(*dest)
                if ag.ref_ij in self.node_table and self.node_table[ag.ref_ij].exists:
                    ref_pos = self.node_table[ag.ref_ij].pos
                else:
                    ref_pos = self._node_pos_from_ij(*ag.ref_ij)
                # Arrival detection: use distance traveled from reference node
                # This is robust to wind drift since it doesn't depend on absolute position
                travel_dist = float(np.linalg.norm(ag.pos - ref_pos))
                if travel_dist >= (self.step_dist * float(self.cfg.arrival_distance_factor)):
                    # New node: create it and assign agent as holder
                    if dest not in self.node_table or not self.node_table[dest].exists:
                        new_node = Node(dest[0], dest[1], dest_pos, self.cfg.phi_init)
                        new_node.time_created = t
                        new_node.agent_id = ag.id
                        self.node_table[dest] = new_node
                        self.nodes_created += 1
                        ag.role = "NODE"
                        ag.held_node = dest
                        ag.ref_ij = dest
                        ag.state = "CIRCLING"
                        ag.orbit_radius = 10.0 + random.uniform(-2.0, 2.0)
                        ag.orbit_center = ag.pos.copy()
                    else:
                        # Existing node: check for replacement opportunity
                        existing_node = self.node_table[dest]
                        ag.ref_ij = dest
                        ag.dest_ij = None
                        ag.branch_sign = None
                        if self.cfg.enable_replacement:
                            # Compute node age (time since creation)
                            age = max(0.0, t - float(existing_node.time_created))
                            if age >= self.cfg.replacement_min_age:
                                # Replacement probability: p = 1 - exp(-(age - min_age) / tau)
                                # Increases exponentially with age, reaching ~63% at min_age + tau
                                tau = max(1e-6, float(self.cfg.replacement_time_scale))
                                p_replace = 1.0 - math.exp(-(age - self.cfg.replacement_min_age) / tau)
                                p_replace = max(0.0, min(1.0, p_replace))
                                if random.random() < p_replace:
                                    pre_drift = self._compute_node_drift(dest)
                                    self.num_swaps += 1
                                    self.swap_times.append(t)
                                    self.swap_nodes.append(dest)
                                    self.swap_ages.append(age)
                                    self.swap_pre_drifts.append(pre_drift)
                                    
                                    if self.cfg.log:
                                        print(f"[t={t:.1f}] Replacement at node {dest}: old={existing_node.agent_id}, new={ag.id}, age={age:.1f}s, p={p_replace:.2f}, pre_drift={pre_drift:.2f}m")
                                    prev_holder_id = existing_node.agent_id
                                    existing_node.agent_id = ag.id
                                    ag.role = "NODE"
                                    ag.held_node = dest
                                    ag.state = "CIRCLING"
                                    prev_holder = None
                                    if prev_holder_id is not None:
                                        for a2 in self.agents:
                                            if a2.id == prev_holder_id:
                                                prev_holder = a2
                                                break
                                    if prev_holder is not None:
                                        ag.orbit_radius = prev_holder.orbit_radius
                                        ag.angular_speed = prev_holder.angular_speed
                                    else:
                                        ag.orbit_radius = 10.0 + random.uniform(-2.0, 2.0)
                                    ag.orbit_center = ag.pos.copy()
                                    
                                    post_drift = self._compute_node_drift(dest)
                                    drift_reduction = pre_drift - post_drift
                                    self.swap_post_drifts.append(post_drift)
                                    self.swap_drift_reductions.append(drift_reduction)
                                    
                                    if self.cfg.log:
                                        print(f"[t={t:.1f}] Post-swap drift at node {dest}: {post_drift:.2f}m, reduction={drift_reduction:.2f}m")
                                    
                                    # Previous holder transitions to HOMING state
                                    # Moves toward new node's orbit center to reduce accumulated drift
                                    if prev_holder is not None and prev_holder.id != ag.id:
                                        prev_holder.held_node = None
                                        prev_holder.role = "ANT"
                                        prev_holder.dest_ij = None
                                        prev_holder.retract_target = None
                                        prev_holder.homing_target = ag.orbit_center.copy()
                                        prev_holder.state = "HOMING"
                                        prev_holder.spiral_r = 0.0
                                        prev_holder.spiral_theta = 0.0
                    ag.travel_start_time = None
                    ag.expected_flight_time = None

        # Recompute hop counts for routing and pheromone updates
        self._recompute_hop_to_base()
        self._recompute_hop_to_user()

        # Count active ants near each node (for pheromone reinforcement)
        ant_counts: Dict[Tuple[int, int], int] = {}
        for ag in self.agents:
            if ag.role == "ANT" and ag.state in ("ACTIVE", "RETRACTING"):
                if ag.ref_ij in self.node_table and self.node_table[ag.ref_ij].exists:
                    if float(np.linalg.norm(ag.pos - self.node_table[ag.ref_ij].pos)) <= self.cfg.comm_r:
                        ant_counts[ag.ref_ij] = ant_counts.get(ag.ref_ij, 0) + 1

        # Data flooding: periodically create new messages at base
        # Uses frontier-based flooding to propagate messages across network
        if t >= self.next_data_time:
            msg_id = f"data_{self.step_idx}_{t:.3f}"
            if (0, 0) in self.node_table:
                base_node = self.node_table[(0, 0)]
                base_node.seen_data_ids[msg_id] = 0
                self.active_data_msgs[msg_id] = {"frontier": {(0, 0)}, "completed": False}
            self.next_data_time = t + self.cfg.data_interval

        # Propagate data messages: frontier expansion at each timestep
        # Each node in frontier forwards to all neighbors within communication range
        for msg_id, info in list(self.active_data_msgs.items()):
            if info.get("completed", False):
                continue
            frontier: Set[Tuple[int, int]] = set(info.get("frontier", set()))
            next_frontier: Set[Tuple[int, int]] = set()
            for key in frontier:
                if key not in self.node_table:
                    continue
                sender = self.node_table[key]
                if not sender.exists or msg_id not in sender.seen_data_ids:
                    continue
                sender_hop = sender.seen_data_ids[msg_id]
                for k2, node2 in self.node_table.items():
                    if not node2.exists:
                        continue
                    if msg_id in node2.seen_data_ids:
                        continue
                    if self._connected(sender.pos, node2.pos):
                        node2.seen_data_ids[msg_id] = sender_hop + 1
                        next_frontier.add(k2)
                        # When message reaches user, record global hop count and start control propagation
                        if float(np.linalg.norm(node2.pos - self.user_pos)) <= self.cfg.comm_r:
                            if msg_id not in self.user_first_hop:
                                self.user_first_hop[msg_id] = sender_hop + 1
                                self.user_received.add(msg_id)
                                # Initialize control message frontier: nodes near user that have seen the data
                                control_frontier = set()
                                for k3, n3 in self.node_table.items():
                                    if n3.exists and msg_id in n3.seen_data_ids and float(np.linalg.norm(n3.pos - self.user_pos)) <= self.cfg.comm_r:
                                        n3.known_global_hopcounts[msg_id] = self.user_first_hop[msg_id]
                                        control_frontier.add(k3)
                                if control_frontier:
                                    self.active_control_msgs[msg_id] = {"frontier": control_frontier, "completed": False, "global_hop": self.user_first_hop[msg_id]}
            if next_frontier:
                info["frontier"] = next_frontier
            else:
                info["completed"] = True

        # Control message propagation: advertise global hop count (base-to-user) to all nodes
        # Allows nodes to determine if they're on the least-hop path
        for data_msg_id, cinfo in list(self.active_control_msgs.items()):
            if cinfo.get("completed", False):
                continue
            frontier: Set[Tuple[int, int]] = set(cinfo.get("frontier", set()))
            next_frontier: Set[Tuple[int, int]] = set()
            global_h = cinfo.get("global_hop", None)
            for key in frontier:
                if key not in self.node_table:
                    continue
                sender = self.node_table[key]
                sender.known_global_hopcounts[data_msg_id] = global_h
                for k2, node2 in self.node_table.items():
                    if not node2.exists:
                        continue
                    if data_msg_id in node2.known_global_hopcounts:
                        continue
                    if self._connected(sender.pos, node2.pos):
                        node2.known_global_hopcounts[data_msg_id] = global_h
                        next_frontier.add(k2)
            if next_frontier:
                cinfo["frontier"] = next_frontier
            else:
                cinfo["completed"] = True

        min_user_hop: Optional[float] = None
        for k, n in self.node_table.items():
            if n.exists and float(np.linalg.norm(n.pos - self.user_pos)) <= self.cfg.comm_r:
                if min_user_hop is None or n.hop_to_base < min_user_hop:
                    min_user_hop = n.hop_to_base

        # Determine which nodes are on least-hop path
        # A node is on least-hop if: hop_to_base + hop_to_user == global_hop_count
        for node in self.node_table.values():
            node.on_least_hop = False

        for k, node in self.node_table.items():
            if not node.exists:
                continue
            for msg_id, global_h in node.known_global_hopcounts.items():
                if node.hop_to_base < float("inf") and node.hop_to_user < float("inf"):
                    if (node.hop_to_base + node.hop_to_user) == global_h:
                        node.on_least_hop = True
                        break

        # Pheromone update: multiple reinforcement mechanisms
        for key, node in list(self.node_table.items()):
            if not node.exists or (node.i == 0 and node.j == 0):
                continue
            # Ant reinforcement: active ants near node increase pheromone
            n_ants = ant_counts.get(key, 0)
            node.phi += self.cfg.phi_ant * n_ants

            # Internal reinforcement: node has children (forward neighbors with agents)
            eligible = False
            for child_ij in [(node.i + 1, node.j), (node.i, node.j + 1)]:
                if child_ij in self.node_table:
                    child_node = self.node_table[child_ij]
                    if child_node.exists and child_node.agent_id is not None:
                        eligible = True
            if eligible:
                node.phi += self.cfg.phi_internal

            # Connection reinforcement: node is on least-hop path to user
            if node.on_least_hop:
                node.phi += self.cfg.phi_conn

            # Decay and clamp to valid range
            node.phi -= self.cfg.phi_decay
            node.phi = max(0.0, min(self.cfg.phi_max, node.phi))

        for k, node in list(self.node_table.items()):
            if node.exists and node.phi <= 1e-6 and k != (0, 0):
                node.exists = False
                node.agent_id = None
                for ag in self.agents:
                    if ag.held_node == k:
                        ag.held_node = None
                        ag.role = "ANT"
                        ag.state = "RETRACTING"
                        ag.dest_ij = None
                        best = None
                        best_phi = -1.0
                        best_sum = 9999
                        for nb in [(node.i - 1, node.j), (node.i, node.j - 1)]:
                            if nb in self.node_table and self.node_table[nb].exists:
                                s = nb[0] + nb[1]
                                phi_val = self.node_table[nb].phi
                                if phi_val > best_phi or (abs(phi_val - best_phi) < 1e-9 and s < best_sum):
                                    best = nb
                                    best_phi = phi_val
                                    best_sum = s
                        ag.ref_ij = k
                        ag.retract_target = best if best is not None else (0, 0)
                        ag.spiral_r = 0.0
                        ag.spiral_theta = 0.0
                        break
                for ag in self.agents:
                    if ag.dest_ij == k:
                        ag.dest_ij = None

        for ag in self.agents:
            if ag.state == "RETRACTING":
                self._agent_step_motion(ag, dt, t)

        for ag in self.agents:
            if ag.state == "ACTIVE" and ag.role == "ANT":
                if ag.ref_ij not in self.node_table or not self.node_table[ag.ref_ij].exists:
                    nearest = None
                    nearest_d = 1e9
                    for k, n in self.node_table.items():
                        if n.exists:
                            d = float(np.linalg.norm(ag.pos - n.pos))
                            if d < nearest_d:
                                nearest = k
                                nearest_d = d
                    if nearest is not None and nearest_d <= 2 * self.cfg.comm_r:
                        ag.ref_ij = nearest

        for ag in self.agents:
            if ag.state != "RETRACTING":
                self._agent_step_motion(ag, dt, t)

        min_d = float("inf")
        for ag in self.agents:
            d = float(np.linalg.norm(ag.pos - self.user_pos))
            if d < min_d:
                min_d = d
        self.min_agent_user_distance = min(self.min_agent_user_distance, min_d)
        self.last_agent_user_distance = min_d
        if not self.success and min_d <= self.cfg.comm_r:
            self.success = True
            self.success_time = t
            min_hop_now = self._min_hop_nodes_to_user()
            self._success_via_min_hop_flag = (min_hop_now is not None)

        self._compute_drift_metric()

        self.t += dt
        self.step_idx += 1

    # ---------------------- subroutines ----------------------
    def _agent_step_motion(self, ag: Agent, dt: float, t: float) -> None:
        """Update agent position and state for one timestep."""
        if ag.state == "ON_GROUND":
            return
        if ag.state == "LAUNCHING":
            if ag.launch_start_time is not None and (t - ag.launch_start_time) >= ag.launch_duration:
                ag.state = "ACTIVE"
                self._agent_choose_branch(ag)
            else:
                return
        if ag.state == "ACTIVE":
            if ag.role == "NODE" and ag.held_node is not None:
                ag.state = "CIRCLING"
                center = ag.orbit_center if ag.orbit_center is not None else self.node_table[ag.held_node].pos
                rel = ag.pos - center
                if float(np.linalg.norm(rel)) < 1e-6:
                    rel = np.array([ag.orbit_radius, 0.0])
                ag.orbit_ang = math.atan2(rel[1], rel[0])
                return
            if ag.dest_ij is None:
                self._agent_choose_branch(ag)
            dest_pos = self._node_pos_from_ij(*ag.dest_ij)
            vec = dest_pos - ag.pos
            dist = float(np.linalg.norm(vec))
            move_dir = self.v_left if ag.branch_sign == +1 else self.v_right
            ag.pos += move_dir * ag.v * dt
            if self.cfg.enable_wind:
                ag.pos += self._get_wind_influence() * dt
            ag.heading = math.atan2(move_dir[1], move_dir[0])
            return

        if ag.state == "CIRCLING":
            if ag.held_node is None:
                ag.state = "ACTIVE"
                ag.role = "ANT"
                return
            center = ag.orbit_center if ag.orbit_center is not None else self.node_table[ag.held_node].pos
            # Orbit center drifts with wind to maintain realistic physics
            if ag.orbit_center is not None and self.cfg.enable_wind:
                ag.orbit_center = ag.orbit_center + self._get_wind_influence() * dt
                center = ag.orbit_center
            # Circular motion: move tangentially, then project back to fixed radius
            rel = ag.pos - center
            r = float(np.linalg.norm(rel))
            if r < 1e-6:
                rel = np.array([ag.orbit_radius, 0.0])
                r = ag.orbit_radius
            # Compute tangent direction (perpendicular to radius vector)
            tangent = np.array([-rel[1], rel[0]])
            tnorm = float(np.linalg.norm(tangent))
            if tnorm > 1e-9:
                tangent_unit = tangent / tnorm
            else:
                tangent_unit = np.array([0.0, 1.0])
            # Move tangentially at constant angular speed
            tangential_speed = ag.angular_speed * max(ag.orbit_radius, r)
            ag.pos += tangent_unit * tangential_speed * dt
            if self.cfg.enable_wind:
                ag.pos += self._get_wind_influence() * dt
            # Project back to fixed radius (maintains circular orbit despite wind)
            rel2 = ag.pos - center
            nrm = float(np.linalg.norm(rel2))
            if nrm > 1e-9:
                ag.pos = center + (rel2 / nrm) * ag.orbit_radius
            ag.orbit_ang = math.atan2(rel[1], rel[0]) + ag.angular_speed * dt
            ag.heading = math.atan2(tangent_unit[1], tangent_unit[0])
            return

        if ag.state == "RETRACTING":
            if ag.retract_target is None:
                candidates: List[Tuple[Tuple[int, int], Node]] = []
                if ag.ref_ij in self.node_table and self.node_table[ag.ref_ij].exists:
                    ref_node = self.node_table[ag.ref_ij]
                    current_sum = ref_node.i + ref_node.j
                    for nb in [(ref_node.i - 1, ref_node.j), (ref_node.i, ref_node.j - 1)]:
                        if nb in self.node_table and self.node_table[nb].exists:
                            nb_node = self.node_table[nb]
                            nb_sum = nb_node.i + nb_node.j
                            if nb_sum < current_sum:
                                candidates.append((nb, nb_node))
                if candidates:
                    best = None
                    best_phi = -1.0
                    for (k2, node2) in candidates:
                        if node2.phi > best_phi:
                            best = k2
                            best_phi = node2.phi
                    ag.retract_target = best
                else:
                    snap_node_key = None
                    snap_node_pos = None
                    min_dist = float("inf")
                    for k2, node2 in self.node_table.items():
                        if node2.exists:
                            d2 = float(np.linalg.norm(ag.pos - node2.pos))
                            if d2 <= self.cfg.comm_r and d2 < min_dist:
                                snap_node_key = k2
                                snap_node_pos = node2.pos
                                min_dist = d2
                    if snap_node_key is None:
                        for other_ag in self.agents:
                            if other_ag.id != ag.id and other_ag.state != "RETRACTING":
                                d3 = float(np.linalg.norm(ag.pos - other_ag.pos))
                                if d3 <= self.cfg.comm_r and d3 < min_dist:
                                    nearest_k = None
                                    nearest_pos = None
                                    nearest_d = float("inf")
                                    for k3, node3 in self.node_table.items():
                                        if node3.exists:
                                            dnode = float(np.linalg.norm(other_ag.pos - node3.pos))
                                            if dnode <= self.cfg.comm_r and dnode < nearest_d:
                                                nearest_k = k3
                                                nearest_pos = node3.pos
                                                nearest_d = dnode
                                    if nearest_k is not None:
                                        snap_node_key = nearest_k
                                        snap_node_pos = nearest_pos
                                        min_dist = d3
                    if snap_node_key is not None and snap_node_pos is not None:
                        vec = snap_node_pos - ag.pos
                        dist = float(np.linalg.norm(vec))
                        if dist > 1e-6:
                            dirv = vec / dist
                            ag.pos += dirv * ag.v * dt
                            if self.cfg.enable_wind:
                                ag.pos += self._get_wind_influence() * dt
                            ag.heading = math.atan2(dirv[1], dirv[0])
                        if dist < 3.0:
                            ag.ref_ij = snap_node_key
                            ag.retract_target = None
                    else:
                        ag.spiral_theta += ag.spiral_omega * dt
                        ag.spiral_r += 0.5 * dt
                        dx = ag.spiral_r * math.cos(ag.spiral_theta)
                        dy = ag.spiral_r * math.sin(ag.spiral_theta)
                        ag.pos += np.array([dx, dy]) * dt
                        if self.cfg.enable_wind:
                            ag.pos += self._get_wind_influence() * dt
                    return
            if ag.retract_target in self.node_table and self.node_table[ag.retract_target].exists:
                dest = self.node_table[ag.retract_target].pos
                vec = dest - ag.pos
                dist = float(np.linalg.norm(vec))
                if dist < 3.0:
                    ag.ref_ij = ag.retract_target
                    ag.retract_target = None
                    if ag.ref_ij == (0, 0):
                        ag.dest_ij = None
                        ag.role = "ANT"
                        ag.state = "ON_GROUND"
                else:
                    ag.pos += (vec / (dist + 1e-12)) * ag.v * dt
                    if self.cfg.enable_wind:
                        ag.pos += self._get_wind_influence() * dt
                    ag.heading = math.atan2(vec[1], vec[0])
            else:
                ag.retract_target = None
            return

        if ag.state == "HOMING":
            if ag.homing_target is None:
                ag.state = "ACTIVE"
                return
            vec = ag.homing_target - ag.pos
            dist = float(np.linalg.norm(vec))
            if dist > 1e-6:
                dirv = vec / dist
                ag.pos += dirv * ag.v * dt
                if self.cfg.enable_wind:
                    ag.pos += self._get_wind_influence() * dt
                ag.heading = math.atan2(dirv[1], dirv[0])
            if dist < 3.0:
                ag.homing_target = None
                ag.state = "ACTIVE"
            return

    def _agent_choose_branch(self, ag: Agent) -> None:
        """Choose next branch for agent using pheromone-based probabilistic selection.
        
        Implements Deneubourg-style decision making:
        - Raw probability based on pheromone levels: p_raw = (mu + phi)^2
        - Geometric bias: favors left branch when closer to base (smaller i+j)
        - Final probability combines both factors
        """
        i, j = ag.ref_ij
        left_ij = (i + 1, j)
        right_ij = (i, j + 1)
        phi_left = self.node_table[left_ij].phi if left_ij in self.node_table else 0.0
        phi_right = self.node_table[right_ij].phi if right_ij in self.node_table else 0.0
        # Raw probability from pheromone (squared to amplify differences)
        pL_raw = (self.cfg.mu + phi_left) ** 2
        pR_raw = (self.cfg.mu + phi_right) ** 2
        # Geometric bias: left branch is closer to base when i < j
        denom = (i + j + 2)
        cL = (i + 1) / denom if denom != 0 else 0.5
        # Normalize pheromone-based probability
        pL = pL_raw / (pL_raw + pR_raw + 1e-12)
        # Combine pheromone probability with geometric bias
        piL = (pL * cL) / (pL * cL + (1 - pL) * (1 - cL) + 1e-12)
        r = random.random()
        if r < piL:
            ag.branch_sign = +1
            ag.dest_ij = left_ij
        else:
            ag.branch_sign = -1
            ag.dest_ij = right_ij
        dest_pos = self._node_pos_from_ij(*ag.dest_ij)
        vec = dest_pos - ag.pos
        if float(np.linalg.norm(vec)) < 1e-6:
            vec = np.array([0.0, 1.0])
        ag.heading = math.atan2(vec[1], vec[0])
        ag.travel_start_time = None
        ag.expected_flight_time = None

    def _recompute_hop_to_base(self) -> None:
        """Recompute hop counts to base using BFS from base node."""
        for node in self.node_table.values():
            node.hop_to_base = float("inf")
        if (0, 0) not in self.node_table:
            return
        from collections import deque

        q = deque()
        self.node_table[(0, 0)].hop_to_base = 0
        q.append((0, 0))
        while q:
            cur = q.popleft()
            cur_node = self.node_table[cur]
            for nb in cur_node.neighbors:
                if nb in self.node_table and self.node_table[nb].exists:
                    if self.node_table[nb].hop_to_base > cur_node.hop_to_base + 1:
                        self.node_table[nb].hop_to_base = cur_node.hop_to_base + 1
                        q.append(nb)

    def _recompute_hop_to_user(self) -> None:
        """Recompute hop counts to user using BFS from nodes near user."""
        for node in self.node_table.values():
            node.hop_to_user = float("inf")
        from collections import deque
        q = deque()
        for k, n in self.node_table.items():
            if n.exists and float(np.linalg.norm(n.pos - self.user_pos)) <= self.cfg.comm_r:
                n.hop_to_user = 0
                q.append(k)
        while q:
            cur = q.popleft()
            cur_node = self.node_table[cur]
            for nb in cur_node.neighbors:
                if nb in self.node_table and self.node_table[nb].exists:
                    if self.node_table[nb].hop_to_user > cur_node.hop_to_user + 1:
                        self.node_table[nb].hop_to_user = cur_node.hop_to_user + 1
                        q.append(nb)

    def _min_hop_nodes_to_user(self) -> Optional[int]:
        """Find minimum hop count among nodes within communication range of user."""
        best: Optional[int] = None
        for k, n in self.node_table.items():
            if n.exists and float(np.linalg.norm(n.pos - self.user_pos)) <= self.cfg.comm_r:
                if n.hop_to_base < float("inf"):
                    hop_val = int(n.hop_to_base)
                    if best is None or hop_val < best:
                        best = hop_val
        return best

    def _compute_node_drift(self, node_key: Tuple[int, int]) -> float:
        """Compute drift for a specific node.
        
        Drift is the Euclidean distance between:
        - Nominal lattice position (ideal position without wind)
        - Current position of the agent holding the node
        
        Returns 0.0 if node doesn't exist or has no holder.
        """
        if node_key not in self.node_table:
            return 0.0
        
        node = self.node_table[node_key]
        if not node.exists or node.agent_id is None:
            return 0.0
        
        # Find the agent holding this node
        holder_agent = None
        for ag in self.agents:
            if ag.id == node.agent_id:
                holder_agent = ag
                break
        
        if holder_agent is None:
            return 0.0
        
        nominal_pos = self._node_pos_from_ij(node.i, node.j)
        drift_distance = float(np.linalg.norm(holder_agent.pos - nominal_pos))
        return drift_distance

    def _compute_drift_metric(self) -> None:
        """Compute instantaneous drift for all nodes with holders.
        
        Accumulates drift samples over time for time-averaged metric.
        Called every timestep to track network geometric stability.
        """
        for node_key, node in self.node_table.items():
            if not node.exists or node.agent_id is None:
                continue
            
            holder_agent = None
            for ag in self.agents:
                if ag.id == node.agent_id:
                    holder_agent = ag
                    break
            
            if holder_agent is None:
                continue
            
            nominal_pos = self._node_pos_from_ij(node.i, node.j)
            drift_distance = float(np.linalg.norm(holder_agent.pos - nominal_pos))
            self.drift_sum += drift_distance
            self.drift_sample_count += 1


# --------------------------
# Optional matplotlib renderer
# --------------------------

class _Renderer:
    """Matplotlib-based renderer for SMAVNET visualization."""
    
    def __init__(self, sim: SmavNet2D):
        """Initialize renderer with simulation reference."""
        self.sim = sim
        self.fig = None
        self.ax = None
        self.next_plot_time = 0.0
        self.plot_every = 0.2

    def setup(self) -> None:
        """Initialize matplotlib figure and axes."""
        import matplotlib.pyplot as plt  # local import to avoid headless penalty

        plt.ion()
        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.ax.set_xlim(0, self.sim.cfg.area_w)
        self.ax.set_ylim(-50, self.sim.cfg.area_h)
        self.ax.set_aspect('equal')
        self.ax.set_title('SMAVNET 2D Simulation', fontsize=14, fontweight='bold')

    def should_draw(self, t: float) -> bool:
        """Check if it's time to redraw the visualization."""
        return t >= self.next_plot_time

    def draw(self, t: float) -> None:
        """Draw current simulation state to matplotlib figure."""
        ax = self.ax
        sim = self.sim
        ax.clear()
        ax.set_xlim(0, sim.cfg.area_w)
        ax.set_ylim(-50, sim.cfg.area_h)
        ax.set_aspect('equal')
        
        user_hop = sim._min_hop_nodes_to_user()
        title = f"SMAVNET Simulation - t={t:.1f}s"
        if user_hop is not None:
            title += f" (User hop={user_hop})"
        if sim.success:
            title += " - SUCCESS!"
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Draw edges (communication links) first
        for k, n in sim.node_table.items():
            if not n.exists:
                continue
            for nb in n.neighbors:
                if nb in sim.node_table and sim.node_table[nb].exists:
                    p1 = n.pos
                    p2 = sim.node_table[nb].pos
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='lightgray', 
                           linewidth=1.0, alpha=0.5, zorder=1)

        # Base and user with better styling
        ax.scatter([sim.base_pos[0]], [sim.base_pos[1]], c='black', s=200, 
                   marker='s', zorder=10, edgecolors='white', linewidths=2)
        ax.scatter([sim.user_pos[0]], [sim.user_pos[1]], c='green', s=300, 
                   marker='*', zorder=10, edgecolors='white', linewidths=2)

        # Nodes with pheromone levels
        node_x, node_y, node_s, node_c = [], [], [], []
        for k, n in sim.node_table.items():
            if n.exists:
                node_x.append(n.pos[0])
                node_y.append(n.pos[1])
                node_s.append(80 + 300 * n.phi)
                node_c.append(n.phi)
        
        if node_x:
            sc = ax.scatter(node_x, node_y, s=node_s, c=node_c, cmap='Reds', 
                           vmin=0.0, vmax=1.0, edgecolors='darkred', linewidths=1.5, 
                           zorder=5, alpha=0.8)
            for k, n in sim.node_table.items():
                if n.exists:
                    if n.on_least_hop:
                        ax.scatter([n.pos[0]], [n.pos[1]], s=200, facecolors='none', 
                                  edgecolors='green', linewidths=2.5, zorder=6)
                    ax.text(n.pos[0], n.pos[1] + 15, f"{n.phi:.3f}", 
                           fontsize=7, color='darkred', ha='center', va='bottom', 
                           zorder=7, bbox=dict(boxstyle='round,pad=0.2', 
                           facecolor='white', alpha=0.7, edgecolor='darkred', linewidth=0.5))

        # Agents with better styling
        agent_x, agent_y = [], []
        agent_colors, agent_sizes = [], []
        
        for ag in sim.agents:
            agent_x.append(ag.pos[0])
            agent_y.append(ag.pos[1])
            
            if ag.state == 'ON_GROUND':
                agent_colors.append('gray')
                agent_sizes.append(50)
            elif ag.state == 'LAUNCHING':
                agent_colors.append('orange')
                agent_sizes.append(60)
            elif ag.state == 'ACTIVE':
                if ag.role == 'ANT':
                    agent_colors.append('blue')
                    agent_sizes.append(60)
                else:
                    agent_colors.append('red')
                    agent_sizes.append(80)
            elif ag.state == 'CIRCLING':
                agent_colors.append('magenta')
                agent_sizes.append(90)
            elif ag.state == 'RETRACTING':
                agent_colors.append('cyan')
                agent_sizes.append(55)
            elif ag.state == 'HOMING':
                agent_colors.append('gold')
                agent_sizes.append(70)
            else:
                agent_colors.append('black')
                agent_sizes.append(40)
        
        for i, (x, y, color, size) in enumerate(zip(agent_x, agent_y, agent_colors, agent_sizes)):
            ax.scatter([x], [y], c=color, s=size, zorder=10, edgecolors='white', linewidths=1.5)
            ax.text(x + 5, y + 5, f"A{sim.agents[i].id}", fontsize=8, 
                   color='black', zorder=11, bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='white', alpha=0.7, edgecolor='none'))
            
            ag = sim.agents[i]
            if ag.dest_ij is not None and ag.state not in ('RETRACTING',):
                dpos = sim._node_pos_from_ij(*ag.dest_ij)
                ax.plot([ag.pos[0], dpos[0]], [ag.pos[1], dpos[1]], 
                       linewidth=1.0, linestyle='--', color='lightgray', 
                       alpha=0.4, zorder=4)
                ax.scatter([dpos[0]], [dpos[1]], marker='x', c='lightgray', 
                          s=40, alpha=0.5, zorder=6)

        # Wind visualization
        if sim.cfg.enable_wind:
            wind_magnitude = float(np.linalg.norm(sim.wind_vector))
            if wind_magnitude > 0.1:
                wind_start = np.array([50.0, sim.cfg.area_h - 50.0])
                wind_end = wind_start + sim.wind_vector * 30.0
                ax.arrow(wind_start[0], wind_start[1], 
                        wind_end[0] - wind_start[0], wind_end[1] - wind_start[1],
                        head_width=15, head_length=12, fc='cyan', ec='cyan', 
                        zorder=13, alpha=0.8, linewidth=2)
                ax.text(wind_start[0] - 30, wind_start[1] + 10, 'Wind', 
                       fontsize=10, color='cyan', fontweight='bold', zorder=13)
                ax.text(wind_start[0] + 40, wind_start[1] + 10, f'{wind_magnitude:.2f} m/s', 
                       fontsize=9, color='cyan', zorder=13)
        
        num_nodes = sum(1 for n in sim.node_table.values() if n.exists)
        num_active = sum(1 for a in sim.agents if a.state == 'ACTIVE')
        num_swaps = getattr(sim, 'num_swaps', 0)
        
        info_text = (
            f"Nodes: {num_nodes}  |  "
            f"Active Agents: {num_active}  |  "
            f"Swaps: {num_swaps}  |  "
            f"Success: {'Yes' if sim.success else 'No'}"
        )
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, 
                         edgecolor='black', linewidth=1.5), zorder=12)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='black', label='Base Station'),
            Patch(facecolor='green', label='User'),
            Patch(facecolor='red', label='Node (pheromone)'),
            Patch(facecolor='blue', label='Ant Agent'),
            Patch(facecolor='red', label='Holder Agent'),
            Patch(facecolor='magenta', label='Circling Agent'),
            Patch(facecolor='cyan', label='Retracting Agent'),
            Patch(facecolor='gold', label='Homing Agent'),
        ]
        if sim.cfg.enable_wind:
            legend_elements.append(Patch(facecolor='cyan', label='Wind'))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                 framealpha=0.9, edgecolor='black')
        
        self.plt.pause(0.01)
        self.next_plot_time = t + self.plot_every


if __name__ == '__main__':
    cfg = SimulationConfig(
        n_agents=8,
        comm_r=100.0,
        duration=1800.0,
        enable_wind=True,
        wind_speed=0.5,
        enable_replacement=True,
        log=False,
        seed=50,
    )
    sim = SmavNet2D(cfg)
    result = sim.run(headless=True)
    
    print(f"Success: {result.success}, Time: {result.success_time:.1f}s, User: {result.user_pos}")
    print(f"Avg Drift: {result.avg_drift:.4f} (raw: {result.avg_drift_raw:.2f}m)")
    if result.num_swaps > 0:
        print(f"Drift Reduction: {result.avg_drift_reduction:.4f} (raw: {result.avg_drift_reduction_raw:.2f}m) per swap ({result.num_swaps} swaps)")
    else:
        print("No swaps occurred")

    sim.reset(seed=50)
    result_viz = sim.run(headless=False)
    print("VISUAL DONE")