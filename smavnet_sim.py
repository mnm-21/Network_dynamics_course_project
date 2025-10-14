import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------
# Public API dataclasses
# --------------------------


@dataclass
class SimulationConfig:
    # Area and timing
    area_w: float = 1000.0
    area_h: float = 1000.0
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
    user_mode: str = "uniform"  # fixed|uniform|gaussian
    user_fixed: Optional[Tuple[float, float]] = None
    user_margin: float = 80.0
    user_gaussian_center: Optional[Tuple[float, float]] = None
    user_gaussian_sigma: float = 120.0

    # Randomness
    seed: Optional[int] = None


@dataclass
class SimulationResult:
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

##CHANGED
class DataPacket:
    def __init__(self, type, data = None, maxHops = 30):
        ## recommended maxHops ~ number of agents. 
        ##type is either BASE or USER , denoting the source of the data packet.
        self.data = data
        self.type = type
        if self.type == "BASE":
            self.hopCounter = 0
        if self.type == "USER":
            self.hopCounter = 1
    def increment(self):
        self.hopCounter+= 1

class Node:
    def __init__(self, i: int, j: int, pos: np.ndarray, phi_init: float):
        self.i = i
        self.j = j
        self.pos = pos.copy()
        self.phi = phi_init
        self.exists = True
        self.hop_to_base = float("inf")
        self.neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
        self.agent_id: Optional[int] = None
        self.time_created: float = 0.0


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

        #commmunication
        self.data_queue = []

    def begin_launch(self, current_time: float, base_pos: np.ndarray):
        if self.state == "ON_GROUND":
            self.state = "LAUNCHING"
            self.launch_start_time = current_time
            self.pos = base_pos.copy() + np.random.normal(scale=0.5, size=2)
            self.ref_ij = (0, 0)
            self.dest_ij = None
            self.branch_sign = None
    
    ## CHANGED
    def get_data(self, data):
        self.data_queue.append(data)
    
    ##CHANGED
    def push_data(self, sim):
        nb_agents = sim.get_nb_agents(sim, self)
        while self.data_queue:
            data = self.data_queue.pop()
            data.increment()

            for agent in nb_agents:
                if data.hopCounter <= data.maxHops:
                    agent.get_data(data)
            #get neighbor agents and do get_data on them and then delete the data from own



class SmavNet2D:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # Geometry
        self.base_pos = np.array([self.cfg.area_w / 2.0, 80.0])
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
        self.launch_times = self._schedule_launches()

        #communication
        self.hopCount = np.inf ##shortest base hopcount encountered by the user so far.

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
        th = math.radians(degrees)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        return R.dot(vec)

    def _node_pos_from_ij(self, i: int, j: int) -> np.ndarray:
        return self.base_pos + self.step_dist * (i * self.v_left + j * self.v_right)

    def _connected(self, a_pos: np.ndarray, b_pos: np.ndarray) -> bool:
        return np.linalg.norm(a_pos - b_pos) <= self.cfg.comm_r + 1e-6

    def _init_nodes(self) -> None:
        base_node = Node(0, 0, self.base_pos, self.cfg.phi_max)
        base_node.hop_to_base = 0
        self.node_table[(0, 0)] = base_node

    def _schedule_launches(self) -> List[float]:
        ints = [max(0.0, random.gauss(self.cfg.launch_mean, self.cfg.launch_jitter)) for _ in range(self.cfg.n_agents)]
        times = []
        acc = 0.0
        for x in ints:
            acc += x
            times.append(acc)
        return times

    def _sample_user_pos(self) -> np.ndarray:
        if self.cfg.user_mode == "fixed" and self.cfg.user_fixed is not None:
            return np.array(self.cfg.user_fixed, dtype=float)
        if self.cfg.user_mode == "gaussian":
            center = (
                np.array(self.cfg.user_gaussian_center, dtype=float)
                if self.cfg.user_gaussian_center is not None
                else np.array([self.cfg.area_w / 2.0, self.cfg.area_h - self.cfg.user_margin])
            )
            x, y = np.random.normal(center[0], self.cfg.user_gaussian_sigma), np.random.normal(center[1], self.cfg.user_gaussian_sigma)
            x = float(np.clip(x, 0.0, self.cfg.area_w))
            y = float(np.clip(y, 0.0, self.cfg.area_h))
            return np.array([x, y])
        # uniform
        margin = self.cfg.user_margin
        x = random.uniform(margin, self.cfg.area_w - margin)
        y = random.uniform(self.cfg.area_h / 2.0, self.cfg.area_h - margin)
        return np.array([x, y])

    # ---------------------- public API ----------------------
    def reset(self, seed: Optional[int] = None) -> None:
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
        self.launch_times = self._schedule_launches()
        self.success = False
        self.success_time = None
        self.min_agent_user_distance = float("inf")
        self.last_agent_user_distance = float("inf")
        self.nodes_created = 1

    def run(self, headless: bool = True, max_time: Optional[float] = None) -> SimulationResult:
        end_time = min(self.cfg.duration, max_time) if max_time is not None else self.cfg.duration
        renderer = None
        if not headless:
            renderer = _Renderer(self)
            renderer.setup()
        while self.t < end_time:
            self.step()
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
        return {
            "t": self.t,
            "num_nodes": float(sum(1 for n in self.node_table.values() if n.exists)),
            "num_agents_launched": float(sum(1 for a in self.agents if a.state != "ON_GROUND")),
            "success": float(1.0 if self.success else 0.0),
        }

    # ---------------------- core step logic ----------------------
    def step(self) -> None:
        t = self.t
        dt = self.dt

        # launch
        for idx, lt in enumerate(self.launch_times):
            if t >= lt and self.agents[idx].state == "ON_GROUND":
                self.agents[idx].begin_launch(t, self.base_pos)

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
                        ag.state = "CIRCLING"
                        ag.orbit_radius = 10.0 + random.uniform(-2.0, 2.0)
                    else:
                        ag.ref_ij = dest
                        ag.dest_ij = None
                        ag.branch_sign = None

        # pheromone updates and hops
        self._recompute_hop_to_base()
        ant_counts: Dict[Tuple[int, int], int] = {}
        for ag in self.agents:
            if ag.role == "ANT" and ag.state in ("ACTIVE", "RETRACTING"):
                if ag.ref_ij in self.node_table and self.node_table[ag.ref_ij].exists:
                    if float(np.linalg.norm(ag.pos - self.node_table[ag.ref_ij].pos)) <= self.cfg.comm_r:
                        ant_counts[ag.ref_ij] = ant_counts.get(ag.ref_ij, 0) + 1

        # minimal-hop among nodes that see user
        min_user_hop: Optional[float] = None
        for k, n in self.node_table.items():
            if n.exists and float(np.linalg.norm(n.pos - self.user_pos)) <= self.cfg.comm_r:
                if min_user_hop is None or n.hop_to_base < min_user_hop:
                    min_user_hop = n.hop_to_base

        for key, node in list(self.node_table.items()):
            if not node.exists or (node.i == 0 and node.j == 0):
                continue
            n_ants = ant_counts.get(key, 0)
            node.phi += self.cfg.phi_ant * n_ants
            # if node.hop_to_base < float("inf"):
            #     node.phi += self.cfg.phi_internal

            i, j = node.i, node.j
            eligible = False
            for child_ij in [(i+1, j), (i, j+1)]:
                if child_ij in self.node_table:
                    child_node = self.node_table[child_ij]
                    if child_node.exists and child_node.agent_id is not None:
                        eligible = True
            if eligible:
                node.phi += self.cfg.phi_internal
                                
            if min_user_hop is not None and node.hop_to_base <= min_user_hop:
                #node.phi += self.cfg.phi_conn
                pass
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
                        # choose neighbor toward base
                        best = None
                        best_sum = 9999
                        for nb in node.neighbors:
                            if nb in self.node_table and self.node_table[nb].exists:
                                s = nb[0] + nb[1]
                                if s < best_sum:
                                    best = nb
                                    best_sum = s
                        ag.ref_ij = best if best is not None else (0, 0)
                        ag.retract_target = None
                        ag.spiral_r = 0.0
                        ag.spiral_theta = 0.0
                        break

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
            if ag.retract_target is None:
                candidates: List[Tuple[Tuple[int, int], Node]] = []
                for k, node in self.node_table.items():
                    if node.exists and float(np.linalg.norm(node.pos - ag.pos)) <= self.cfg.comm_r * 2.0:
                        candidates.append((k, node))
                if candidates:
                    best = None
                    best_score = -1e9
                    cur_ref_sum = sum(ag.ref_ij) if ag.ref_ij else 9999
                    for (k, node) in candidates:
                        gain = cur_ref_sum - (k[0] + k[1])
                        score = node.phi + 0.5 * max(gain, 0)
                        if score > best_score:
                            best_score = score
                            best = k
                    ag.retract_target = best
                else:
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
                else:
                    ag.pos += (vec / (dist + 1e-12)) * ag.v * dt
                    ag.heading = math.atan2(vec[1], vec[0])
            else:
                ag.retract_target = None
            return

    def _agent_choose_branch(self, ag: Agent) -> None:
        i, j = ag.ref_ij
        left_ij = (i + 1, j)
        right_ij = (i, j + 1)
        phi_left = self.node_table[left_ij].phi if left_ij in self.node_table else 0.0
        phi_right = self.node_table[right_ij].phi if right_ij in self.node_table else 0.0
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

    def _min_hop_nodes_to_user(self) -> Optional[int]:
        best: Optional[int] = None
        for k, n in self.node_table.items():
            if n.exists and float(np.linalg.norm(n.pos - self.user_pos)) <= self.cfg.comm_r:
                if n.hop_to_base < float("inf"):
                    hop_val = int(n.hop_to_base)
                    if best is None or hop_val < best:
                        best = hop_val
        return best

    ## CHANGED
    def get_nb_agents(self, agent):
        node = self.node_table[agent.held_node]
        nb_nodes = node.neighbors
        res = []
        for node_ in nb_nodes:
            if node_ in self.node_table and node_.agent_id:
                res.append(self.agents[node_.agent_id])
        return res

## TODO implement the broadcasting of the data packaets by the base and user, calling the push_data functions in the agents, and handlign when data reaches the user and updating the hopcount global thing
#then use it to check if agent is on the shortest path and update pheromone accordingly. 

# --------------------------
# Optional matplotlib renderer
# --------------------------

class _Renderer:
    def __init__(self, sim: SmavNet2D):
        self.sim = sim
        self.fig = None
        self.ax = None
        self.next_plot_time = 0.0
        self.plot_every = 0.2

    def setup(self) -> None:
        import matplotlib.pyplot as plt  # local import to avoid headless penalty

        plt.ion()
        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(6, 9))
        self.ax.set_xlim(0, self.sim.cfg.area_w)
        self.ax.set_ylim(0, self.sim.cfg.area_h)
        self.ax.set_aspect('equal')
        self.ax.set_title('SMAVNET 2D (API)')

    def should_draw(self, t: float) -> bool:
        return t >= self.next_plot_time

    def draw(self, t: float) -> None:
        ax = self.ax
        sim = self.sim
        ax.clear()
        ax.set_xlim(0, sim.cfg.area_w)
        ax.set_ylim(0, sim.cfg.area_h)
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
            if ag.dest_ij is not None:
                dpos = sim._node_pos_from_ij(*ag.dest_ij)
                ax.scatter([dpos[0]], [dpos[1]], marker='x', c='black', s=30, zorder=6)
                ax.plot([ag.pos[0], dpos[0]], [ag.pos[1], dpos[1]], linewidth=0.8, linestyle='--', color='0.4', zorder=4)

        #annotate the phi values on the nodes
        for k,n in sim.node_table.items():
            if n.exists:
                ax.text(n.pos[0]+60, n.pos[1], f"{n.phi:.5f}", fontsize=6, color='k', ha='center', va='center', zorder=6)


        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        info = (
            f"Nodes: {sum(1 for n in sim.node_table.values() if n.exists)}  |  "
            f"Agents active: {sum(1 for a in sim.agents if a.state=='ACTIVE')}"
        )
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6), zorder=12)
        self.plt.pause(0.001)
        self.next_plot_time = t + self.plot_every


