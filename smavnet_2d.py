# save as smavnet_2d.py
import math
import time
import random
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt

## TODO implementation of commmunication - must be done once ~5-10*dt. data packet can be implemented as a separate datastructure. 
## TODO package the code into multiple files, and handle the global variables better. This is required for the experimentation.
#can implement like a queue in each agent to model the asynchronous manner of communication and delays.

## TODO retraction graphics is not handle properly
# --------------------------
# Simulation parameters
# --------------------------
AREA_W = 1000.0
AREA_H = 1000.0
BASE_POS = np.array([AREA_W / 2.0, 80.0])    # base near bottom center
USER_POS = np.array([AREA_W / 2.0, AREA_H - 600.0])  # user near top center (fixed for single trial)
DT = 0.1                     # simulation step (s)
TRIAL_DURATION = 900.0       # seconds (15 min) - single trial
V = 10.0                     # agent speed (m/s)
COMM_R = 100.0               # communication radius (m)
N_AGENTS = 18                # total number of MAVs available (launched sequentially)
LAUNCH_MEAN = 15.0           # mean inter-launch (s)
LAUNCH_JITTER = 7.5          # jitter for inter-launch (s)
LAUNCH_DURATION = 4.0        # seconds spent in LAUNCHING state

# Pheromone parameters (from paper-ish values)
PHI_INIT = 0.7
PHI_MAX = 1.0
PHI_ANT = 0.0001               # per ant "visit" per dt (scaled by dt in code)
PHI_CONN = 0.01              # reinforcement for being on least-hop path (per dt)
PHI_INTERNAL = 0.001
PHI_DECAY = 0.001            # per dt evaporation (we will scale by dt)
MU = 0.75                    # exploration constant for Deneubourg model
EPS = 0.01                   # small tie-breaker in selection

# --------------------------
# TODO: lattice directions are defined in terms of the user location, which is an unknown. Needs to be defined absolutely. 


# Geometry for i,j lattice
# Choose base orientation: vector pointing from base to user
## NOTE changed the base dir to an absolute north direction from being user-relative.
base_to_user = USER_POS - BASE_POS
north_dir = np.array([0.0, 1.0])
base_dir = north_dir / (np.linalg.norm(north_dir) + 1e-9)
# Two lattice basis vectors at +30° (left) and -30° (right) from base_dir
def rotate(vec, degrees):
    th = math.radians(degrees)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s],[s, c]])
    return R.dot(vec)

V_LEFT = rotate(base_dir, +30.0)   # i direction
V_RIGHT = rotate(base_dir, -30.0)  # j direction
STEP_DIST = COMM_R * 0.95          # spacing between adjacent lattice nodes (~comm range)
# Node location from indices (i,j):
def node_pos_from_ij(i,j):
    return BASE_POS + STEP_DIST * (i * V_LEFT + j * V_RIGHT)

# --------------------------
# Data structures
# --------------------------
class Node:
    """A node stored by its lattice coordinates (i,j)."""
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.pos = node_pos_from_ij(i,j)
        self.phi = PHI_INIT
        self.exists = True
        self.hop_to_base = np.inf   # min-hop to base (integer)
        self.neighbors = []         # list of neighbor (i,j) coords (adjacent lattice)
        self.agent_id = None        # ID of agent currently holding this node (if any)
        self.time_created = 0.0
        # populate neighbors (4-neighborhood in i,j lattice: (i+1,j), (i-1,j), (i,j+1), (i,j-1))
        self.neighbors = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]

class Agent:
    """An agent that can be ANT (explorer), NODE (holding), RETRACT (returning), or ON_GROUND/LAUNCHING."""
    def __init__(self, aid):
        self.id = aid
        self.pos = BASE_POS.copy() + np.random.normal(scale=1.0, size=2)
        self.heading = random.uniform(0, 2*math.pi)
        self.v = V
        self.state = 'ON_GROUND'     # ON_GROUND, LAUNCHING, ACTIVE, CIRCLING (node's orbit), RETRACTING
        self.role = 'ANT'            # ANT or NODE (when agent is the one physically holding a node it is CIRCLING and role=NODE)
        self.launch_start_time = None
        self.launch_duration = LAUNCH_DURATION
        # reference node coordinate (i,j) from which this ant was launched
        self.ref_ij = (0,0)          # base node initial
        # chosen destination lattice index (i',j') for current exploration step
        self.dest_ij = None
        # branch angle: +30 or -30 in radians (relative to base_dir)
        self.branch_sign = None
        # if agent is circling as a node:
        self.orbit_radius = 10.0     # meters
        self.orbit_ang = 0.0
        self.angular_speed = 0.8     # rad/s for circling
        # retraction specifics
        self.retract_target = None   # neighbor (i,j) to aim for when retracting
        self.spiral_theta = 0.0
        self.spiral_r = 0.0
        self.spiral_omega = 2.0      # rad/s
        # bookkeeping: which node (i,j) if any is currently reserved by this agent
        self.held_node = None

    def begin_launch(self, current_time):
        if self.state == 'ON_GROUND':
            self.state = 'LAUNCHING'
            self.launch_start_time = current_time
            # keep pos at base (maybe slight offset random)
            self.pos = BASE_POS.copy() + np.random.normal(scale=0.5, size=2)
            # reset reference to base
            self.ref_ij = (0,0)
            self.dest_ij = None
            self.branch_sign = None

    def complete_launch(self, node_table):
        self.state = 'ACTIVE'
        # choose a branch immediately from the reference node (pass current node_table)
        self.choose_branch(node_table)

    def choose_branch(self, node_table=None):
        """Choose left/right branch (i+1,j) or (i,j+1) based on Deneubourg formula using phi of child candidates."""
        i,j = self.ref_ij
        left_ij = (i+1, j)
        right_ij = (i, j+1)
        # gather phi's from node_table (if child exists) else 0.0
        phi_left = node_table[left_ij].phi if left_ij in node_table else 0.0
        phi_right = node_table[right_ij].phi if right_ij in node_table else 0.0
        pL_raw = (MU + phi_left)**2
        pR_raw = (MU + phi_right)**2
        # correction factor cL = (i+1)/(i+j+2) as in paper
        denom = (i + j + 2)
        if denom == 0:
            cL = 0.5
        else:
            cL = (i + 1) / denom
        pL = pL_raw / (pL_raw + pR_raw + 1e-12)
        # corrected piL:
        piL = (pL * cL) / (pL * cL + (1-pL) * (1-cL) + 1e-12)
        # choose
        r = random.random()
        if r < piL:
            self.branch_sign = +1   # left
            self.dest_ij = left_ij
        else:
            self.branch_sign = -1   # right
            self.dest_ij = right_ij
        # set heading vector to point to destination position
        dest_pos = node_pos_from_ij(*self.dest_ij)
        vec = dest_pos - self.pos
        if np.linalg.norm(vec) < 1e-6:
            # if we already at dest, orientation along base_dir
            vec = base_dir.copy()
        self.heading = math.atan2(vec[1], vec[0])

    def step_motion(self, dt, t, node_table):
        """Advance agent state and motion depending on role/state."""
        if self.state == 'ON_GROUND':
            return
        if self.state == 'LAUNCHING':
            if (t - self.launch_start_time) >= self.launch_duration:
                self.complete_launch(node_table)
            else:
                return
        if self.state == 'ACTIVE':
            # if this agent is currently assigned as node holder, switch to circling
            if self.role == 'NODE' and self.held_node is not None:
                self.state = 'CIRCLING'
                # position should be near node's pos
                n = node_table[self.held_node]
                self.pos = n.pos + np.array([self.orbit_radius, 0.0])
                self.orbit_ang = 0.0
                return
            # if ant with a destination, move straight toward dest_pos
            if self.dest_ij is None:
                # choose a branch now
                self.choose_branch(node_table)
            dest_pos = node_pos_from_ij(*self.dest_ij)
            vec = dest_pos - self.pos
            dist = np.linalg.norm(vec)
            if dist < 3.0:   # arrived near destination zone
                # check if node exists here
                if self.dest_ij in node_table and node_table[self.dest_ij].exists:
                    # set that node as reference and continue outward
                    self.ref_ij = self.dest_ij
                    self.dest_ij = None
                    # small pause & choose next branch next step
                else:
                    # become the NODE at this dest
                    # Node creation handled externally by main loop to assign Node object and link agent
                    pass
            else:
                # advance straight
                dirv = vec / (dist + 1e-12)
                self.pos += dirv * self.v * dt
                self.heading = math.atan2(dirv[1], dirv[0])
            return

        if self.state == 'CIRCLING':
            # orbit around held node
            if self.held_node is None:
                # safety: revert to ACTIVE
                self.state = 'ACTIVE' ## NOTE held_node is None => pheromone depleted and node removed; shouldnt it become RETRACTING instead
                self.role = 'ANT'
                return
            node = node_table[self.held_node]
            # increment angle
            self.orbit_ang += self.angular_speed * dt
            # position relative to node pos
            self.pos = node.pos + np.array([self.orbit_radius * math.cos(self.orbit_ang),
                                            self.orbit_radius * math.sin(self.orbit_ang)])
            # maintain heading tangent
            tangent = np.array([-math.sin(self.orbit_ang), math.cos(self.orbit_ang)])
            self.heading = math.atan2(tangent[1], tangent[0])
            return

        if self.state == 'RETRACTING':
            # deterministic: choose neighbor with highest phi that is closer to base (i+j smaller)
            # if agent has no current retract target or target disappeared, choose new
            if self.retract_target is None:
                # find neighbors of current nearest node or nearby nodes within COMM_R
                # pick the node among node_table keys within COMM_R with highest phi and (i+j) smaller than current ref
                
                candidates = []
                ## TODO candidates are all 4 neighbors, not the 2 in the backward direction
                ## just implement as (i-1,j) and (i, j-1)??? Why use COMM_R anol?
                for k, node in node_table.items():
                    if node.exists and np.linalg.norm(node.pos - self.pos) <= COMM_R * 2.0:
                        candidates.append((k, node))
                if candidates:
                    # prefer those with smaller (i+j) (closer to base) and higher phi
                    # score = phi + alpha*(delta_improvement)
                    best = None
                    best_score = -1e9
                    cur_ref_sum = sum(self.ref_ij) if self.ref_ij else 9999
                    for (k, node) in candidates:
                        gain = cur_ref_sum - (k[0] + k[1])
                        score = node.phi + 0.5 * max(gain, 0)
                        if score > best_score:
                            best_score = score
                            best = k
                    self.retract_target = best
                else:
                    # no candidates nearby => spiral to try reconnect
                    self.spiral_theta += self.spiral_omega * dt
                    self.spiral_r += 0.5 * dt   # slow outward spiral radius
                    # move along spiral in local coordinates
                    dx = self.spiral_r * math.cos(self.spiral_theta)
                    dy = self.spiral_r * math.sin(self.spiral_theta)
                    # small displacement from current pos
                    self.pos += np.array([dx, dy]) * dt
                    return
            # if we have a retract_target, move straight to it
            if self.retract_target in node_table and node_table[self.retract_target].exists:
                dest = node_table[self.retract_target].pos
                vec = dest - self.pos
                dist = np.linalg.norm(vec)
                if dist < 3.0:
                    # arrived to that node; now update ref_ij and continue retracting toward base
                    self.ref_ij = self.retract_target
                    # reset target; we'll choose next closer neighbor in next iteration
                    self.retract_target = None
                else:
                    self.pos += (vec / (dist + 1e-12)) * self.v * dt
                    self.heading = math.atan2(vec[1], vec[0])
            else:
                # target disappeared; reset and try spiral
                ## NOTE spiralling must be based on t_lost, not this. This also means current implementation is not positionless. 
                self.retract_target = None
            return

# --------------------------
# Simulation state: node table and agent list
# --------------------------
node_table = {}   # key: (i,j) -> Node object
agents = [Agent(aid=i) for i in range(N_AGENTS)]

# create base node at (0,0) and mark as special (base)
base_node = Node(0,0)
base_node.pos = BASE_POS.copy()
base_node.phi = PHI_MAX
base_node.hop_to_base = 0
base_node.agent_id = None
node_table[(0,0)] = base_node

# user phantom node (not a permanent node but for hop calc we may add U-hop when user is in range)
# We'll treat user as external; nodes within COMM_R of user will be considered contact nodes to mark path.

# Prepare launch times cumulative
launch_intervals = [max(0.0, random.gauss(LAUNCH_MEAN, LAUNCH_JITTER/2.0)) for _ in range(N_AGENTS)]
launch_times = []
cum = 0.0
for x in launch_intervals:
    cum += x
    launch_times.append(cum)

# --------------------------
# Helper functions
# --------------------------
def connected(a_pos, b_pos):
    return np.linalg.norm(a_pos - b_pos) <= COMM_R + 1e-6

def recompute_hop_to_base(node_table):
    # simple BFS from base (0,0)
    for node in node_table.values():
        node.hop_to_base = np.inf
    if (0,0) not in node_table:
        return
    q = deque()
    node_table[(0,0)].hop_to_base = 0
    q.append((0,0))
    while q:
        cur = q.popleft()
        cur_node = node_table[cur]
        for nb in cur_node.neighbors:
            if nb in node_table and node_table[nb].exists:
                if node_table[nb].hop_to_base > cur_node.hop_to_base + 1:
                    node_table[nb].hop_to_base = cur_node.hop_to_base + 1
                    q.append(nb)

def nodes_in_comm_range_of_user(node_table):
    res = []
    for k,n in node_table.items():
        if n.exists and np.linalg.norm(n.pos - USER_POS) <= COMM_R:
            res.append(k)
    return res

# --------------------------
# Plotting setup (live)
# --------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(6,9))
ax.set_xlim(0, AREA_W)
ax.set_ylim(0, AREA_H)
ax.set_aspect('equal')
ax.set_title('SMAVNET 2D simplified interactive trial')
base_scatter = ax.scatter([BASE_POS[0]], [BASE_POS[1]], c='black', s=80, marker='s', label='Base')
user_scatter = ax.scatter([USER_POS[0]], [USER_POS[1]], c='green', s=100, marker='*', label='User')
agent_scatter = None
node_scat = None
conn_lines = []

# --------------------------
# Simulation main loop (single trial)
# --------------------------
t = 0.0
step = 0
next_plot_time = 0.0
PLOT_EVERY = 0.2   # seconds between redraws (for performance)
created_nodes = set([(0,0)])

print("Starting single trial simulation. Press Ctrl+C to stop early.")
try:
    while t < TRIAL_DURATION:
        # launch logic
        for idx, lt in enumerate(launch_times):
            if t >= lt and agents[idx].state == 'ON_GROUND':
                agents[idx].begin_launch(t)
                print(f"[{t:.1f}s] Agent {agents[idx].id} BEGIN LAUNCH")

        # agent step motions
        for ag in agents:
            ag.step_motion(DT, t, node_table)

        # detect arrivals: for any ACTIVE ant near its dest that is not yet a node, create node and assign agent as holder
        ## NOTE why is this not handled in the agent step_motion?
        for ag in agents:
            if ag.state == 'ACTIVE' and ag.dest_ij is not None:
                dest = ag.dest_ij
                dest_pos = node_pos_from_ij(*dest)
                if np.linalg.norm(ag.pos - dest_pos) < 4.0:
                    # arrival
                    if dest not in node_table or not node_table[dest].exists:
                        # create node and assign to this agent
                        new_node = Node(dest[0], dest[1])
                        new_node.pos = dest_pos
                        new_node.time_created = t
                        new_node.agent_id = ag.id
                        new_node.phi = PHI_INIT #needs to be redefined here for the case of re-activation of agent after retraction
                        node_table[dest] = new_node
                        created_nodes.add(dest)
                        # agent becomes node holder (circling)
                        ag.role = 'NODE'
                        ag.held_node = dest
                        ag.state = 'CIRCLING'
                        ag.orbit_radius = 10.0 + random.uniform(-2.0,2.0)
                        print(f"[{t:.1f}s] Agent {ag.id} BECOMES NODE at {dest}")
                    else:
                        # node exists: become referencing that node
                        ag.ref_ij = dest
                        ag.dest_ij = None
                        ag.branch_sign = None
                        # slight pause before selecting next branch
        
        # pheromone deposition: for each node, count ants in range (ants referencing that node)
        ant_counts = defaultdict(int)
        for ag in agents:
            if ag.role == 'ANT' and ag.state in ('ACTIVE','RETRACTING'):
                # if within COMM_R of some node, count it as referencing that node if it's their ref_ij
                if ag.ref_ij in node_table and node_table[ag.ref_ij].exists:
                    if np.linalg.norm(ag.pos - node_table[ag.ref_ij].pos) <= COMM_R:
                        ant_counts[ag.ref_ij] += 1
        # apply pheromone updates per node
        # recompute hop counts first
        recompute_hop_to_base(node_table)
        # detect which nodes are on least-hop path to user (if any node within COMM_R of user exists, get its hop count and mark upward)
        user_nodes = nodes_in_comm_range_of_user(node_table)
        # determine minimal hop among nodes that see user
        min_user_hop = min([node_table[u].hop_to_base for u in user_nodes]) if user_nodes else None
        # apply updates
        for key, node in node_table.items():
            if not node.exists or (node.i == 0 and node.j == 0):
                continue
            # deposit from ants in range
            n_ants = ant_counts.get(key, 0)
            node.phi += PHI_ANT * n_ants ## NOTE this is causing the pheromone saturation issue. pheromone saturates as soon as another agent uses the node as reference node. 

            # # reinforcement if node is on a route (simple condition: if hop_to_base finite and node is closer to base than some neighbor)
            # ## TODO phi_int is implemented wrong. Should be based on the existence of nodes farther from base.
            # if node.hop_to_base < np.inf:
            #     node.phi += PHI_INTERNAL

            # reinforcement: increase phi by PHI_INTERNAL only if (i+1, j) or (i, j+1) node exists and is held by an agent
            i, j = node.i, node.j
            eligible = False
            for child_ij in [(i+1, j), (i, j+1)]:
                if child_ij in node_table:
                    child_node = node_table[child_ij]
                    if child_node.exists and child_node.agent_id is not None:
                        eligible = True
            if eligible:
                node.phi += PHI_INTERNAL
                
            # if node is on least-hop path to user (we approximate: if min_user_hop exists and node.hop_to_base <= min_user_hop)
            if min_user_hop is not None and node.hop_to_base <= min_user_hop:
                #node.phi += PHI_CONN
                pass
                ## TODO this also seems fishy. This can only
            # evaporation
            node.phi -= PHI_DECAY
            # clamp
            node.phi = max(0.0, min(PHI_MAX, node.phi))

        # Node -> Ant reversion when phi decays to zero
        for k, node in list(node_table.items()):
            if node.exists and node.phi <= 1e-6 and k != (0,0):
                # free this node and cause the holding agent to retract
                holder = node.agent_id
                node.exists = False
                node.agent_id = None
                print(f"[{t:.1f}s] Node {k} PHI depleted -> removed; becomes ANT again")
                # find agent that was holding it
                for ag in agents:
                    if ag.held_node == k:
                        ag.held_node = None
                        ag.role = 'ANT'
                        ag.state = 'RETRACTING'
                        # set its reference to nearest existing node (prefer parent towards base)
                        # simple: pick neighbor node with smallest (i+j)
                        neighbors = node.neighbors
                        best = None
                        best_sum = 9999
                        for nb in neighbors:
                            if nb in node_table and node_table[nb].exists:
                                s = nb[0] + nb[1]
                                if s < best_sum:
                                    best = nb
                                    best_sum = s
                        if best is not None:
                            ag.ref_ij = best
                        else:
                            ag.ref_ij = (0,0)
                        ag.retract_target = None
                        ag.spiral_r = 0.0
                        ag.spiral_theta = 0.0
                        # continue as RETRACTING
                        break

        # connectivity edges for plotting / and hop updates use existing nodes only
        # also detect if user is connected via any chain (if any node within COMM_R of user and hop finite)
        user_connected = False
        for k in node_table:
            if node_table[k].exists and np.linalg.norm(node_table[k].pos - USER_POS) <= COMM_R:
                # if node has finite hop_to_base, then path established
                if node_table[k].hop_to_base < np.inf:
                    user_connected = True
                    break

        # advance time and let retracting agents act
        for ag in agents:
            if ag.state == 'RETRACTING':
                ag.step_motion(DT, t, node_table)

        # small policy: ants far away without reference reassign to nearest existing node if any
        for ag in agents:
            if ag.state == 'ACTIVE' and ag.role == 'ANT':
                if ag.ref_ij not in node_table or not node_table[ag.ref_ij].exists:
                    # choose nearest existing node within 2*COMM_R else keep moving
                    nearest = None
                    nearest_d = 1e9
                    for k,n in node_table.items():
                        if n.exists:
                            d = np.linalg.norm(ag.pos - n.pos)
                            if d < nearest_d:
                                nearest = k
                                nearest_d = d
                    if nearest is not None and nearest_d <= 2*COMM_R:
                        ag.ref_ij = nearest

        # update agent motions for non-retracting ones (some were handled earlier in step_motion)
        for ag in agents:
            if ag.state not in ('RETRACTING',):
                ag.step_motion(DT, t, node_table)

        # drawing / plotting
                # drawing / plotting (clean environment view)
        if t >= next_plot_time:
            ax.clear()
            ax.set_xlim(0, AREA_W)
            ax.set_ylim(0, AREA_H)
            ax.set_aspect('equal')
            ax.set_title(f"t={t:.1f}s  user_connected={user_connected}")
            # draw grid faintly for orientation (optional)
            ax.grid(False)

            # draw base and user
            ax.scatter([BASE_POS[0]], [BASE_POS[1]], c='black', s=100, marker='s', label='Base')
            ax.scatter([USER_POS[0]], [USER_POS[1]], c='green', s=120, marker='*', label='User')

            # draw existing nodes as circles sized by phi (no colorbar)
            node_x = []
            node_y = []
            node_s = []
            node_c = []
            for k,n in node_table.items():
                if n.exists:
                    node_x.append(n.pos[0])
                    node_y.append(n.pos[1])
                    node_s.append(50 + 250 * n.phi)   # reasonable sizes
                    node_c.append(n.phi)
            if node_x:
                sc = ax.scatter(node_x, node_y, s=node_s, c=node_c, cmap='Reds', vmin=0.0, vmax=1.0,
                                edgecolors='k', zorder=5)
                # small legend patch for node marker
                ax.scatter([], [], c='red', s=40, label='Node (phi)')

            # draw edges between neighboring nodes (light gray)
            for k,n in node_table.items():
                if not n.exists:
                    continue
                for nb in n.neighbors:
                    if nb in node_table and node_table[nb].exists:
                        p1 = n.pos; p2 = node_table[nb].pos
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='lightgray', linewidth=0.7, zorder=1)
            
            #annotate the phi values on the nodes
            for k,n in node_table.items():
                if n.exists:
                    ax.text(n.pos[0]+60, n.pos[1], f"{n.phi:.5f}", fontsize=6, color='k', ha='center', va='center', zorder=6)

            # draw agents with colors by state and small labels for id
            xs = [ag.pos[0] for ag in agents]
            ys = [ag.pos[1] for ag in agents]
            colors = []
            sizes = []
            for ag in agents:
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

            # annotate agent ids near markers (small)
            for ag in agents:
                ax.text(ag.pos[0] + 4.0, ag.pos[1] + 2.0, f"{ag.id}", fontsize=7, color='k', zorder=11)

            # draw each agent's destination (if set) as a small X and connect with a faint line
            for ag in agents:
                if ag.dest_ij is not None:
                    dpos = node_pos_from_ij(*ag.dest_ij)
                    ax.scatter([dpos[0]], [dpos[1]], marker='x', c='black', s=30, zorder=6)
                    ax.plot([ag.pos[0], dpos[0]], [ag.pos[1], dpos[1]], linewidth=0.8, linestyle='--', color='0.4', zorder=4)

            # draw reference node markers (small translucent circles)
            for ag in agents:
                if ag.ref_ij in node_table and node_table[ag.ref_ij].exists:
                    rpos = node_table[ag.ref_ij].pos
                    ax.scatter([rpos[0]], [rpos[1]], s=20, facecolors='none', edgecolors='0.2', alpha=0.5, zorder=7)

            # legend
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

            # compact info text
            info = f"Nodes: {sum(1 for n in node_table.values() if n.exists)}  |  Agents active: {sum(1 for a in agents if a.state=='ACTIVE')}"
            ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.6), zorder=12)

            plt.pause(0.001)
            next_plot_time = t + PLOT_EVERY

        # advance global time
        t += DT
        step += 1

    print("Trial complete.")
    plt.ioff()
    plt.show()

except KeyboardInterrupt:
    print("Simulation interrupted by user. Showing final state.")
    plt.ioff()
    plt.show()