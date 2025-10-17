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
    step_dist_factor: float = 0.95  # times comm_r

    # User placement
    user_mode: str = "uniform"  # fixed|uniform
    user_fixed: Optional[Tuple[float, float]] = None
    user_margin: float = 50.0
    user_gaussian_center: Optional[Tuple[float, float]] = None
    user_gaussian_sigma: float = 120.0

    # Communication / data flood
    data_interval: float = 6.0  # seconds between base data floods

    # Randomness
    seed: Optional[int] = None

    # termination conditions
    terminate_on_success: bool = False


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
        self.hop_to_user = float("inf")  # UHop
        self.neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
        self.agent_id: Optional[int] = None
        self.time_created: float = 0.0

        # Communication bookkeeping
        self.seen_data_ids: Dict[str, int] = {}  # msg_id -> hop_count seen for data flooding
        self.known_global_hopcounts: Dict[str, int] = {}  # data msg_id -> global hopcount learned via control propagation
        self.on_least_hop: bool = False  # set each timestep if node believes it's on least-hop route for any msg


class Agent:
    def __init__(self, agent_id: int, base_pos: np.ndarray, speed: float, launch_duration: float):
        self.id = agent_id
        self.pos = base_pos.copy() + np.random.normal(scale=1.0, size=2)
        self.heading = random.uniform(0.0, 2.0 * math.pi)
        self.v = speed
        self.state = "ON_GROUND"  # ON_GROUND, LAUNCHING, ACTIVE, CIRCLING, RETRACTING
        self.role = "ANT"  # ANT or NODE
        self.launch_start_time: Optional[float] = None
        self.launch_duration = launch_duration
        self.ref_ij = (0, 0)
        self.dest_ij: Optional[Tuple[int, int]] = None
        self.branch_sign: Optional[int] = None
        # circling
        self.orbit_radius = 10.0
        self.orbit_ang = 0.0
        self.angular_speed = 0.8
        # retracting
        self.retract_target: Optional[Tuple[int, int]] = None
        self.spiral_theta = 0.0
        self.spiral_r = 0.0
        self.spiral_omega = 2.0
        self.held_node: Optional[Tuple[int, int]] = None

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

        # lattice base direction: true north (y+ axis), independent of user
        base_dir = np.array([0.0, 1.0])
        self.v_left = self._rotate(base_dir, +self.cfg.lattice_angle_deg)
        self.v_right = self._rotate(base_dir, -self.cfg.lattice_angle_deg)
        self.step_dist = self.cfg.comm_r * self.cfg.step_dist_factor

        # state
        self.t = 0.0
        self.step_idx = 0
        self.node_table: Dict[Tuple[int, int], Node] = {}
        self.agents: List[Agent] = [
            Agent(i, self.base_pos, self.cfg.speed, self.cfg.launch_duration)
            for i in range(self.cfg.n_agents)
        ]
        self._init_nodes()
        # continuous launch scheduler: attempt a launch every sampled interval
        self.next_launch_time = self._sample_launch_interval()

        # communication / data flood bookkeeping
        self.next_data_time = 0.0
        self.active_data_msgs: Dict[str, Dict] = {}  # msg_id -> {'frontier': set(keys), 'completed': bool}
        self.active_control_msgs: Dict[str, Dict] = {}  # data_msg_id -> {'frontier': set(keys), 'completed': bool}
        self.user_first_hop: Dict[str, int] = {}  # msg_id -> hop_count (first-arrival)
        self.user_received: Set[str] = set()

        # metrics
        self.success = False
        self.success_time: Optional[float] = None
        self.min_agent_user_distance = float("inf")
        self.last_agent_user_distance = float("inf")
        self.nodes_created = 1  # base node
        self._success_via_min_hop_flag: Optional[bool] = None

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

        # Their perpendicular outward directions (unit normals)
        left_perp = self._rotate(left_dir, -90.0)   # outward to left
        right_perp = self._rotate(right_dir, +90.0) # outward to right

        # Each offset line passes through (base + comm_r * perp)
        left_origin = np.array([base_x, base_y]) + comm_r * left_perp
        right_origin = np.array([base_x, base_y]) + comm_r * right_perp

        min_y = comm_r * math.cos(angle_rad) * 2
        max_y = self.cfg.area_h - self.cfg.user_margin

        y = random.uniform(min_y, max_y)

        # For each offset line, find x at this y (since y = origin_y + t*dy → t = (y - origin_y)/dy)
        t_left = (y - left_origin[1]) / left_dir[1]
        t_right = (y - right_origin[1]) / right_dir[1]

        x_left = left_origin[0] + t_left * left_dir[0]
        x_right = right_origin[0] + t_right * right_dir[0]
        x = random.uniform(x_left, x_right)

        return np.array([x, y])

    # ---------------------- public API ----------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset simulation state for a new run with optional new seed."""
        if seed is not None:
            self.cfg.seed = seed
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
        # reinit
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
        # compute end-of-run hop diagnostics
        min_hop_end = self._min_hop_nodes_to_user()
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
        """Advance simulation by one timestep."""
        t = self.t
        dt = self.dt

        # periodic launch attempts: every ~launch_mean±jitter seconds, try to launch one ON_GROUND agent at base
        if t >= self.next_launch_time:
            launched = False
            for ag in self.agents:
                if ag.state == "ON_GROUND":
                    if float(np.linalg.norm(ag.pos - self.base_pos)) <= self.cfg.comm_r:
                        ag.begin_launch(t, self.base_pos)
                        launched = True
                        break
            # schedule next attempt regardless of success
            self.next_launch_time = t + self._sample_launch_interval()

        # agent motions (first pass)
        for ag in self.agents:
            self._agent_step_motion(ag, dt, t)

        # arrivals and node creation
        for ag in self.agents:
            if ag.state == "ACTIVE" and ag.dest_ij is not None:
                dest = ag.dest_ij
                dest_pos = self._node_pos_from_ij(*dest)
                if float(np.linalg.norm(ag.pos - dest_pos)) < 4.0:
                    if dest not in self.node_table or not self.node_table[dest].exists:
                        new_node = Node(dest[0], dest[1], dest_pos, self.cfg.phi_init)
                        new_node.time_created = t
                        new_node.agent_id = ag.id
                        self.node_table[dest] = new_node
                        self.nodes_created += 1
                        ag.role = "NODE"
                        ag.held_node = dest
                        ag.ref_ij = dest  # update reference to the new node
                        ag.state = "CIRCLING"
                        ag.orbit_radius = 10.0 + random.uniform(-2.0, 2.0)
                    else:
                        ag.ref_ij = dest
                        ag.dest_ij = None
                        ag.branch_sign = None

        # === recompute hops ===
        self._recompute_hop_to_base()
        self._recompute_hop_to_user()  # new: compute hop_to_user (UHop) via BFS from nodes near user

        # ant counts (agents referencing nodes)
        ant_counts: Dict[Tuple[int, int], int] = {}
        for ag in self.agents:
            if ag.role == "ANT" and ag.state in ("ACTIVE", "RETRACTING"):
                if ag.ref_ij in self.node_table and self.node_table[ag.ref_ij].exists:
                    if float(np.linalg.norm(ag.pos - self.node_table[ag.ref_ij].pos)) <= self.cfg.comm_r:
                        ant_counts[ag.ref_ij] = ant_counts.get(ag.ref_ij, 0) + 1

        # --- DATA FLOODING: maybe create a new data message at base periodically ---
        if t >= self.next_data_time:
            msg_id = f"data_{self.step_idx}_{t:.3f}"
            # start flooding from base (base node exists)
            if (0, 0) in self.node_table:
                base_node = self.node_table[(0, 0)]
                base_node.seen_data_ids[msg_id] = 0
                self.active_data_msgs[msg_id] = {"frontier": {(0, 0)}, "completed": False}
            self.next_data_time = t + self.cfg.data_interval

        # propagate data floods (synchronous per-timestep frontier expansion)
        for msg_id, info in list(self.active_data_msgs.items()):
            if info.get("completed", False):
                continue
            frontier: Set[Tuple[int, int]] = set(info.get("frontier", set()))
            next_frontier: Set[Tuple[int, int]] = set()
            # for each node in frontier, forward to any node within comm range that hasn't seen the msg
            for key in frontier:
                if key not in self.node_table:
                    continue
                sender = self.node_table[key]
                if not sender.exists or msg_id not in sender.seen_data_ids:
                    continue
                sender_hop = sender.seen_data_ids[msg_id]
                # consider all other nodes in the table (could be optimized with spatial index)
                for k2, node2 in self.node_table.items():
                    if not node2.exists:
                        continue
                    if msg_id in node2.seen_data_ids:
                        continue
                    # if node2 is within comm range of sender (spatial connectivity)
                    if self._connected(sender.pos, node2.pos):
                        node2.seen_data_ids[msg_id] = sender_hop + 1
                        next_frontier.add(k2)
                        # if node2 is within comm range of user, user receives the data via this node
                        if float(np.linalg.norm(node2.pos - self.user_pos)) <= self.cfg.comm_r:
                            # record first arrival at user for this msg if not already recorded
                            if msg_id not in self.user_first_hop:
                                self.user_first_hop[msg_id] = sender_hop + 1
                                self.user_received.add(msg_id)
                                # initialize control propagation frontier with nodes that are within user range and have seen the msg
                                # any such node acts as a gateway to advertise the discovered global hopcount
                                control_frontier = set()
                                for k3, n3 in self.node_table.items():
                                    if n3.exists and msg_id in n3.seen_data_ids and float(np.linalg.norm(n3.pos - self.user_pos)) <= self.cfg.comm_r:
                                        n3.known_global_hopcounts[msg_id] = self.user_first_hop[msg_id]
                                        control_frontier.add(k3)
                                # start control propagation
                                if control_frontier:
                                    self.active_control_msgs[msg_id] = {"frontier": control_frontier, "completed": False, "global_hop": self.user_first_hop[msg_id]}
            # update frontier
            if next_frontier:
                info["frontier"] = next_frontier
            else:
                info["completed"] = True

        # propagate control adverts (global hopcount) across nodes (flood until all learn)
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
                # sender should already have the known_global_hopcounts entry but ensure it
                sender.known_global_hopcounts[data_msg_id] = global_h
                # advertise to any node within comm range that doesn't yet know
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

        # minimal-hop among nodes that see user
        min_user_hop: Optional[float] = None
        for k, n in self.node_table.items():
            if n.exists and float(np.linalg.norm(n.pos - self.user_pos)) <= self.cfg.comm_r:
                if min_user_hop is None or n.hop_to_base < min_user_hop:
                    min_user_hop = n.hop_to_base

        # --- determine which nodes are on least-hop route (using learned global hopcounts) ---
        # reset on_least_hop flags
        for node in self.node_table.values():
            node.on_least_hop = False

        # for every known global hopcount that a node has learned, the node can check if B+U == global_h
        for k, node in self.node_table.items():
            if not node.exists:
                continue
            for msg_id, global_h in node.known_global_hopcounts.items():
                # only mark on_least_hop if node.hop_to_base and hop_to_user are finite
                if node.hop_to_base < float("inf") and node.hop_to_user < float("inf"):
                    if (node.hop_to_base + node.hop_to_user) == global_h:
                        node.on_least_hop = True
                        # once true for one msg, we can stop checking other msgs for this node
                        break

        # pheromone update loop (use new communication-driven flags)
        for key, node in list(self.node_table.items()):
            if not node.exists or (node.i == 0 and node.j == 0):
                continue
            n_ants = ant_counts.get(key, 0)
            node.phi += self.cfg.phi_ant * n_ants

            # internal reinforcement if node has at least one neighbor that is farther from base (approx)
            eligible = False
            for child_ij in [(node.i + 1, node.j), (node.i, node.j + 1)]:
                if child_ij in self.node_table:
                    child_node = self.node_table[child_ij]
                    if child_node.exists and child_node.agent_id is not None:
                        eligible = True
            if eligible:
                node.phi += self.cfg.phi_internal

            # connect / least-hop reinforcement
            if node.on_least_hop:
                node.phi += self.cfg.phi_conn

            # decay and clamp
            node.phi -= self.cfg.phi_decay
            node.phi = max(0.0, min(self.cfg.phi_max, node.phi))

        # Node -> Ant reversion when phi decays to zero (except base)
        for k, node in list(self.node_table.items()):
            if node.exists and node.phi <= 1e-6 and k != (0, 0):
                holder = node.agent_id
                node.exists = False
                node.agent_id = None
                # find agent that was holding it
                for ag in self.agents:
                    if ag.held_node == k:
                        ag.held_node = None
                        ag.role = "ANT"
                        ag.state = "RETRACTING"
                        ag.dest_ij = None  # clear any plotted destination X/line
                        # choose backward neighbor only: (i-1,j) or (i,j-1)
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
                        ag.ref_ij = k  # keep current position as reference
                        ag.retract_target = best if best is not None else (0, 0)
                        ag.spiral_r = 0.0
                        ag.spiral_theta = 0.0
                        break
                # Clear any agents that had this node as a destination so the X disappears
                for ag in self.agents:
                    if ag.dest_ij == k:
                        ag.dest_ij = None

        # retracting motions
        for ag in self.agents:
            if ag.state == "RETRACTING":
                self._agent_step_motion(ag, dt, t)

        # reassign references when needed
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

        # final per-step motions for non-retracting
        for ag in self.agents:
            if ag.state != "RETRACTING":
                self._agent_step_motion(ag, dt, t)

        # metrics and success detection (success: any agent within comm_r of user)
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
            # If at this time there exists at least one node within user range with finite hop,
            # mark success_via_min_hop_flag True; else False.
            min_hop_now = self._min_hop_nodes_to_user()
            self._success_via_min_hop_flag = (min_hop_now is not None)

        # advance time
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
                n = self.node_table[ag.held_node]
                ag.pos = n.pos + np.array([ag.orbit_radius, 0.0])
                ag.orbit_ang = 0.0
                return
            if ag.dest_ij is None:
                self._agent_choose_branch(ag)
            dest_pos = self._node_pos_from_ij(*ag.dest_ij)
            vec = dest_pos - ag.pos
            dist = float(np.linalg.norm(vec))
            if dist < 3.0:
                # arrival handled by main loop
                pass
            else:
                dirv = vec / (dist + 1e-12)
                ag.pos += dirv * ag.v * dt
                ag.heading = math.atan2(dirv[1], dirv[0])
            return

        if ag.state == "CIRCLING":
            if ag.held_node is None:
                ag.state = "ACTIVE"
                ag.role = "ANT"
                return
            node = self.node_table[ag.held_node]
            ag.orbit_ang += ag.angular_speed * dt
            ag.pos = node.pos + np.array([
                ag.orbit_radius * math.cos(ag.orbit_ang),
                ag.orbit_radius * math.sin(ag.orbit_ang),
            ])
            tangent = np.array([-math.sin(ag.orbit_ang), math.cos(ag.orbit_ang)])
            ag.heading = math.atan2(tangent[1], tangent[0])
            return

        if ag.state == "RETRACTING":
            # choose between two immediate backward neighbors: (i-1,j) and (i,j-1)
            if ag.retract_target is None:
                candidates: List[Tuple[Tuple[int, int], Node]] = []
                if ag.ref_ij in self.node_table and self.node_table[ag.ref_ij].exists:
                    ref_node = self.node_table[ag.ref_ij]
                    # find backward neighbors: (i-1,j) and (i,j-1) only
                    current_sum = ref_node.i + ref_node.j
                    for nb in [(ref_node.i - 1, ref_node.j), (ref_node.i, ref_node.j - 1)]:
                        if nb in self.node_table and self.node_table[nb].exists:
                            nb_node = self.node_table[nb]
                            nb_sum = nb_node.i + nb_node.j
                            # only consider neighbors closer to base (smaller i+j)
                            if nb_sum < current_sum:
                                candidates.append((nb, nb_node))
                if candidates:
                    # choose the one with higher pheromone
                    best = None
                    best_phi = -1.0
                    for (k2, node2) in candidates:
                        if node2.phi > best_phi:
                            best = k2
                            best_phi = node2.phi
                    ag.retract_target = best
                else:
                    # no backward neighbors available, spiral search
                    # attempt to reconnect to nearest node (preferred) or via nearby agent's node
                    snap_node_key = None
                    snap_node_pos = None
                    min_dist = float("inf")
                    # prefer existing nodes within comm range
                    for k2, node2 in self.node_table.items():
                        if node2.exists:
                            d2 = float(np.linalg.norm(ag.pos - node2.pos))
                            if d2 <= self.cfg.comm_r and d2 < min_dist:
                                snap_node_key = k2
                                snap_node_pos = node2.pos
                                min_dist = d2
                    # if none, try via nearby non-retracting agent by snapping to their nearest node
                    if snap_node_key is None:
                        for other_ag in self.agents:
                            if other_ag.id != ag.id and other_ag.state != "RETRACTING":
                                d3 = float(np.linalg.norm(ag.pos - other_ag.pos))
                                if d3 <= self.cfg.comm_r and d3 < min_dist:
                                    # find nearest existing node to that agent within comm range
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
                        # move toward the snap node
                        vec = snap_node_pos - ag.pos
                        dist = float(np.linalg.norm(vec))
                        if dist > 1e-6:
                            dirv = vec / dist
                            ag.pos += dirv * ag.v * dt
                            ag.heading = math.atan2(dirv[1], dirv[0])
                        # when close enough, snap and resume backward selection next step
                        if dist < 3.0:
                            ag.ref_ij = snap_node_key
                            ag.retract_target = None
                    else:
                        # continue spiral if no targets found
                        ag.spiral_theta += ag.spiral_omega * dt
                        ag.spiral_r += 0.5 * dt
                        dx = ag.spiral_r * math.cos(ag.spiral_theta)
                        dy = ag.spiral_r * math.sin(ag.spiral_theta)
                        ag.pos += np.array([dx, dy]) * dt
                    return
            if ag.retract_target in self.node_table and self.node_table[ag.retract_target].exists:
                dest = self.node_table[ag.retract_target].pos
                vec = dest - ag.pos
                dist = float(np.linalg.norm(vec))
                if dist < 3.0:
                    ag.ref_ij = ag.retract_target
                    ag.retract_target = None
                    # if returned to base, clear any previous plotting state and mark ON_GROUND for relaunch
                    if ag.ref_ij == (0, 0):
                        ag.dest_ij = None
                        ag.role = "ANT"
                        ag.state = "ON_GROUND"
                else:
                    ag.pos += (vec / (dist + 1e-12)) * ag.v * dt
                    ag.heading = math.atan2(vec[1], vec[0])
            else:
                ag.retract_target = None
            return

    def _agent_choose_branch(self, ag: Agent) -> None:
        """Choose next branch for agent using pheromone-based probabilistic selection."""
        i, j = ag.ref_ij
        left_ij = (i + 1, j)
        right_ij = (i, j + 1)
        phi_left = self.node_table[left_ij].phi if left_ij in self.node_table else 0.0
        phi_right = self.node_table[right_ij].phi if right_ij in self.node_table else 0.0
        # Deneubourg-style probabilistic branch selection
        pL_raw = (self.cfg.mu + phi_left) ** 2
        pR_raw = (self.cfg.mu + phi_right) ** 2
        denom = (i + j + 2)
        cL = (i + 1) / denom if denom != 0 else 0.5
        pL = pL_raw / (pL_raw + pR_raw + 1e-12)
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
            # orient along true north when already at destination
            vec = np.array([0.0, 1.0])
        ag.heading = math.atan2(vec[1], vec[0])

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
        # BFS seeded from nodes that are within comm range of user. Those nodes have hop_to_user=0
        for node in self.node_table.values():
            node.hop_to_user = float("inf")
        from collections import deque
        q = deque()
        # seed nodes within user comm range
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
        self.fig, self.ax = plt.subplots(figsize=(6, 9))
        self.ax.set_xlim(0, self.sim.cfg.area_w)
        self.ax.set_ylim(0, self.sim.cfg.area_h)
        self.ax.set_aspect('equal')
        self.ax.set_title('SMAVNET 2D (API)')

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
        ax.set_title(f"t={t:.1f}s  success={sim.success}  user_hop={user_hop}")

        # base and user
        ax.scatter([sim.base_pos[0]], [sim.base_pos[1]], c='black', s=100, marker='s', label='Base')
        ax.scatter([sim.user_pos[0]], [sim.user_pos[1]], c='green', s=120, marker='*', label='User')

        # nodes
        node_x, node_y, node_s, node_c = [], [], [], []
        for k, n in sim.node_table.items():
            if n.exists:
                node_x.append(n.pos[0])
                node_y.append(n.pos[1])
                node_s.append(50 + 250 * n.phi)
                node_c.append(n.phi)
        if node_x:
            sc = ax.scatter(node_x, node_y, s=node_s, c=node_c, cmap='Reds', vmin=0.0, vmax=1.0,
                            edgecolors='k', zorder=5)
            ax.scatter([], [], c='red', s=40, label='Node (phi)')

        # edges
        for k, n in sim.node_table.items():
            if not n.exists:
                continue
            for nb in n.neighbors:
                if nb in sim.node_table and sim.node_table[nb].exists:
                    p1 = n.pos
                    p2 = sim.node_table[nb].pos
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='lightgray', linewidth=0.7, zorder=1)

        # agents
        xs = [ag.pos[0] for ag in sim.agents]
        ys = [ag.pos[1] for ag in sim.agents]
        colors, sizes = [], []
        for ag in sim.agents:
            if ag.state == 'ON_GROUND':
                colors.append('gray'); sizes.append(40)
            elif ag.state == 'LAUNCHING':
                colors.append('orange'); sizes.append(50)
            elif ag.state == 'ACTIVE':
                if ag.role == 'ANT':
                    colors.append('blue'); sizes.append(40)
                else:
                    colors.append('red'); sizes.append(60)
            elif ag.state == 'CIRCLING':
                colors.append('magenta'); sizes.append(70)
            elif ag.state == 'RETRACTING':
                colors.append('cyan'); sizes.append(45)
            else:
                colors.append('black'); sizes.append(35)
        ax.scatter(xs, ys, c=colors, s=sizes, zorder=10)

        # annotate ids and dests
        for ag in sim.agents:
            ax.text(ag.pos[0] + 4.0, ag.pos[1] + 2.0, f"{ag.id}", fontsize=7, color='k', zorder=11)
            # show i,j coordinates
            ax.text(ag.pos[0] + 4.0, ag.pos[1] - 8.0, f"({ag.ref_ij[0]},{ag.ref_ij[1]})", fontsize=6, color='blue', zorder=11)
            if ag.dest_ij is not None and ag.state not in ('RETRACTING',):
                dpos = sim._node_pos_from_ij(*ag.dest_ij)
                ax.scatter([dpos[0]], [dpos[1]], marker='x', c='black', s=30, zorder=6)
                ax.plot([ag.pos[0], dpos[0]], [ag.pos[1], dpos[1]], linewidth=0.8, linestyle='--', color='0.4', zorder=4)

        # annotate the phi values on the nodes
        for k,n in sim.node_table.items():
            if n.exists:
                ax.text(n.pos[0]+60, n.pos[1], f"{n.phi:.5f}", fontsize=6, color='k', ha='center', va='center', zorder=6)
                # show hop metrics near node for debugging
                ax.text(n.pos[0]-40, n.pos[1], f"B{int(n.hop_to_base) if n.hop_to_base<1e8 else '-'}/U{int(n.hop_to_user) if n.hop_to_user<1e8 else '-'}", fontsize=6, color='k', ha='center', va='center', zorder=6)
                if n.on_least_hop:
                    ax.scatter([n.pos[0]], [n.pos[1]], s=140, facecolors='none', edgecolors='green', linewidths=1.2, zorder=7)

        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        info = (
            f"Nodes: {sum(1 for n in sim.node_table.values() if n.exists)}  |  "
            f"Agents active: {sum(1 for a in sim.agents if a.state=='ACTIVE')}"
        )
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6), zorder=12)
        self.plt.pause(0.01)
        self.next_plot_time = t + self.plot_every


if __name__ == '__main__':
    cfg = SimulationConfig(
        n_agents=8,
        comm_r=100.0,
        duration=1800.0,
        speed=10.0,
        seed=42,             # optional for determinism
        user_mode="uniform", # fixed|uniform
        data_interval=6.0,
    )
    sim = SmavNet2D(cfg)
    # Headless fast run to the configured duration (records first success time)
    result = sim.run(headless=True)
    print("HEADLESS:", result.success, result.success_time, result.user_pos)
    # Visual run
    sim.reset(seed=42)
    result_viz = sim.run(headless=False)
    print("VISUAL DONE")