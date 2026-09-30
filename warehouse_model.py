"""
warehouse_model.py
Core modeling, ABC slotting heuristic, QUBO formulation, Quantum-Inspired Simulated Annealing solver,
and order simulation metrics for a 20x20 (400-slot) warehouse with 100 SKUs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import numpy as np

GRID_SIZE = 20  # 20x20 = 400 slots
TOTAL_SLOTS = GRID_SIZE * GRID_SIZE
NUM_SKUS = 100
DEPOT_COORDS = (0, 0)  # Input/Output loading dock at bottom-left corner

CATEGORIES = [
    {"name": "Electronics & Gadgets", "color": "#EF4444"},
    {"name": "Tools & Hardware", "color": "#F59E0B"},
    {"name": "Health & Personal Care", "color": "#10B981"},
    {"name": "Apparel & Accessories", "color": "#8B5CF6"},
    {"name": "Home & Kitchen", "color": "#06B6D4"},
]

@dataclass
class SKU:
    id: int
    name: str
    category: str
    abc_class: str  # 'A', 'B', or 'C'
    velocity: float  # Pick frequency / popularity score
    pick_count: int
    weight: float

@dataclass
class WarehouseEnvironment:
    grid_size: int = GRID_SIZE
    skus: List[SKU] = field(default_factory=list)
    affinity_matrix: np.ndarray = field(default_factory=lambda: np.zeros((NUM_SKUS, NUM_SKUS)))
    simulated_orders: List[List[int]] = field(default_factory=list)
    slot_coords: List[Tuple[int, int]] = field(default_factory=list)
    dist_matrix: np.ndarray = field(default_factory=lambda: np.zeros((TOTAL_SLOTS, TOTAL_SLOTS)))
    depot_dist: np.ndarray = field(default_factory=lambda: np.zeros(TOTAL_SLOTS))


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Standard warehouse orthogonal aisle distance."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def generate_warehouse_data(seed: int = 42, num_orders: int = 1200) -> WarehouseEnvironment:
    """
    Generates deterministic, realistic 100 SKUs with Pareto/ABC velocity,
    category clusters, co-occurrence affinity, and order histories.
    """
    rng = np.random.default_rng(seed)
    env = WarehouseEnvironment()

    # 1. Precompute Slot Coordinates and Distance Matrices
    # Slot index s = y * GRID_SIZE + x
    env.slot_coords = [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)]
    
    # Depot distance
    env.depot_dist = np.array([manhattan_distance(coord, DEPOT_COORDS) for coord in env.slot_coords], dtype=np.float64)

    # Pairwise slot distance matrix (400 x 400)
    coords_arr = np.array(env.slot_coords)  # (400, 2)
    diff_x = np.abs(coords_arr[:, 0, None] - coords_arr[:, 0])
    diff_y = np.abs(coords_arr[:, 1, None] - coords_arr[:, 1])
    env.dist_matrix = (diff_x + diff_y).astype(np.float64)

    # 2. Generate 100 SKUs with Pareto ABC velocity
    # Pareto distribution: 20% A (70% volume), 30% B (20% volume), 50% C (10% volume)
    raw_velocities = rng.pareto(a=1.35, size=NUM_SKUS) + 1.0
    sorted_indices = np.argsort(-raw_velocities)
    normalized_vels = raw_velocities[sorted_indices]
    
    # Scale to realistic daily pick frequencies (total ~ 10,000 picks)
    normalized_vels = (normalized_vels / np.sum(normalized_vels)) * 10000.0

    skus = []
    for idx in range(NUM_SKUS):
        if idx < 20:
            abc = 'A'
        elif idx < 50:
            abc = 'B'
        else:
            abc = 'C'
        
        cat_idx = (idx % len(CATEGORIES))
        cat = CATEGORIES[cat_idx]["name"]
        
        sku = SKU(
            id=idx,
            name=f"SKU-{idx+1:03d}",
            category=cat,
            abc_class=abc,
            velocity=float(round(normalized_vels[idx], 1)),
            pick_count=int(round(normalized_vels[idx])),
            weight=float(round(rng.uniform(0.5, 12.0), 1))
        )
        skus.append(sku)
    env.skus = skus

    # 3. Generate Co-occurrence Affinity Matrix (Item Bundles & Correlations)
    affinity = np.zeros((NUM_SKUS, NUM_SKUS), dtype=np.float64)
    
    for i in range(NUM_SKUS):
        for j in range(i + 1, NUM_SKUS):
            same_cat = (env.skus[i].category == env.skus[j].category)
            pair_affinity = 0.0
            if same_cat:
                if rng.random() < 0.50:
                    pair_affinity += rng.uniform(10.0, 45.0)
            
            # Specific high-affinity cross-category "power pairs" (frequently bundled)
            if (i % 7 == j % 7) and rng.random() < 0.40:
                pair_affinity += rng.uniform(20.0, 55.0)

            # High velocity pairs get naturally more co-picks
            vel_factor = np.sqrt(env.skus[i].velocity * env.skus[j].velocity) / 30.0
            pair_affinity *= (0.5 + 0.5 * vel_factor)
            
            affinity[i, j] = pair_affinity
            affinity[j, i] = pair_affinity
    
    # 4. Generate Simulated Orders based on Velocities & Affinity
    orders = []
    sku_probs = normalized_vels / np.sum(normalized_vels)

    for _ in range(num_orders):
        order_len = int(rng.choice([2, 3, 4, 5], p=[0.40, 0.32, 0.20, 0.08]))
        order_items = []
        
        first_item = int(rng.choice(NUM_SKUS, p=sku_probs))
        order_items.append(first_item)
        
        for _ in range(order_len - 1):
            aff_scores = np.zeros(NUM_SKUS)
            for existing in order_items:
                aff_scores += affinity[existing]
            
            combined_weights = 0.75 * aff_scores + 0.25 * (sku_probs * 1000)
            for existing in order_items:
                combined_weights[existing] = 0.0
            
            total_w = np.sum(combined_weights)
            if total_w > 0:
                p_items = combined_weights / total_w
                next_item = int(rng.choice(NUM_SKUS, p=p_items))
                order_items.append(next_item)
            else:
                break
        
        orders.append(sorted(order_items))
    
    # Empirical affinity matrix from the actual simulated orders
    empirical_affinity = np.zeros((NUM_SKUS, NUM_SKUS), dtype=np.float64)
    for order in orders:
        for i_idx in range(len(order)):
            for j_idx in range(i_idx + 1, len(order)):
                u, v = order[i_idx], order[j_idx]
                empirical_affinity[u, v] += 1.0
                empirical_affinity[v, u] += 1.0
    
    env.affinity_matrix = empirical_affinity
    env.simulated_orders = orders
    return env


def solve_abc_slotting(env: WarehouseEnvironment) -> Dict[int, int]:
    """
    Classical ABC Slotting Heuristic (Before Warehouse):
    - Pure velocity-based allocation.
    - Sort slots strictly by distance from Depot (0, 0).
    - Sort SKUs by ABC class & velocity (A first, then B, then C).
    - Assign top-velocity SKUs to closest slots to Depot.
    - Leaves outer slots empty.
    - Completely ignores item-to-item affinity!
    """
    slot_dist = env.depot_dist.copy()
    sorted_slots = np.argsort(slot_dist)

    sku_to_slot = {}
    for idx in range(NUM_SKUS):
        sku_to_slot[idx] = int(sorted_slots[idx])

    return sku_to_slot


def compute_qubo_energy(
    assignment: Dict[int, int],
    env: WarehouseEnvironment,
    w_depot: float = 1.0,
    w_affinity: float = 1.8
) -> Tuple[float, float, float]:
    """
    Evaluates the Quadratic Hamiltonian Cost:
    H = w_depot * sum_i (v_i * d(pi(i), depot)) + w_affinity * sum_{i < j} (A_ij * d(pi(i), pi(j)))
    """
    depot_cost = 0.0
    for sku_id, slot_id in assignment.items():
        v = env.skus[sku_id].velocity
        d = env.depot_dist[slot_id]
        depot_cost += v * d
    
    sku_ids = list(assignment.keys())
    slots = np.array([assignment[i] for i in sku_ids])
    
    assigned_dist = env.dist_matrix[np.ix_(slots, slots)]
    sub_aff = env.affinity_matrix[np.ix_(sku_ids, sku_ids)]
    
    triu_indices = np.triu_indices(len(sku_ids), k=1)
    affinity_cost = float(np.sum(sub_aff[triu_indices] * assigned_dist[triu_indices]))
    
    total_energy = w_depot * depot_cost + w_affinity * affinity_cost
    return float(total_energy), float(depot_cost), float(affinity_cost)


def solve_qubo_slotting_simulated_annealing(
    env: WarehouseEnvironment,
    w_depot: float = 1.0,
    w_affinity: float = 1.8,
    max_steps: int = 15000,
    initial_temp: float = 2000.0,
    cooling_rate: float = 0.9996,
    seed: int = 42
) -> Tuple[Dict[int, int], List[float]]:
    """
    Quantum-Inspired Simulated Annealing Solver for the Quadratic Assignment Problem (QUBO):
    
    Minimizes:
        H(x) = w_depot * sum_i (v_i * d(slot_i, depot)) + w_affinity * sum_{i < j} (A_ij * d(slot_i, slot_j))
    
    Subject to:
        Each SKU is placed in exactly 1 slot, no two SKUs share a slot.
    
    Features:
    - Affinity-targeted swap and relocation moves to accelerate convergence.
    - Delta-energy calculation for fast O(degree) state swaps.
    - Warm-start from ABC slotting to guarantee strict Pareto improvement.
    """
    rng = np.random.default_rng(seed)
    
    current_assignment = solve_abc_slotting(env)
    
    slot_to_sku = np.full(TOTAL_SLOTS, -1, dtype=int)
    for sku_id, slot_id in current_assignment.items():
        slot_to_sku[slot_id] = sku_id

    empty_slots = [s for s in range(TOTAL_SLOTS) if slot_to_sku[s] == -1]

    velocities = np.array([env.skus[i].velocity for i in range(NUM_SKUS)], dtype=np.float64)
    depot_dist = env.depot_dist
    dist_mat = env.dist_matrix
    aff_mat = env.affinity_matrix

    current_energy, _, _ = compute_qubo_energy(current_assignment, env, w_depot, w_affinity)
    best_energy = current_energy
    
    sku_slots = np.array([current_assignment[i] for i in range(NUM_SKUS)], dtype=int)
    best_slots = sku_slots.copy()
    
    energy_history = [best_energy]
    temp = initial_temp

    for step in range(max_steps):
        move_dice = rng.random()
        
        if move_dice < 0.45:
            # Affinity-targeted swap: pick item i, then pick candidate j close to i's affinity partner
            i = rng.integers(0, NUM_SKUS)
            partners = np.where(aff_mat[i] > 3)[0]
            if len(partners) > 0 and rng.random() < 0.70:
                p = rng.choice(partners)
                p_slot = sku_slots[p]
                # Pick a slot within distance 3 of partner
                d_to_p = dist_mat[p_slot]
                nearby_cand = [s for s in np.where((d_to_p <= 3) & (d_to_p > 0))[0] if slot_to_sku[s] != -1 and slot_to_sku[s] != i]
                if nearby_cand:
                    s_chosen = rng.choice(nearby_cand)
                    j = slot_to_sku[s_chosen]
                else:
                    j = rng.integers(0, NUM_SKUS)
                    while j == i: j = rng.integers(0, NUM_SKUS)
            else:
                j = rng.integers(0, NUM_SKUS)
                while j == i: j = rng.integers(0, NUM_SKUS)
            
            s_i = sku_slots[i]
            s_j = sku_slots[j]
            
            delta_depot = (velocities[i] * (depot_dist[s_j] - depot_dist[s_i]) +
                           velocities[j] * (depot_dist[s_i] - depot_dist[s_j]))
            
            diff_dist_i = dist_mat[s_j, sku_slots] - dist_mat[s_i, sku_slots]
            diff_dist_j = dist_mat[s_i, sku_slots] - dist_mat[s_j, sku_slots]
            delta_aff_k = (aff_mat[i] * diff_dist_i + aff_mat[j] * diff_dist_j)
            delta_aff_k[i] = 0.0
            delta_aff_k[j] = 0.0
            delta_affinity = np.sum(delta_aff_k)
            
            delta_energy = w_depot * delta_depot + w_affinity * delta_affinity

            if delta_energy < 0 or (temp > 1e-4 and rng.random() < np.exp(-delta_energy / temp)):
                sku_slots[i] = s_j
                sku_slots[j] = s_i
                slot_to_sku[s_i] = j
                slot_to_sku[s_j] = i
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_slots = sku_slots.copy()

        elif move_dice < 0.75:
            # Move to empty slot near affinity partner
            i = rng.integers(0, NUM_SKUS)
            s_i = sku_slots[i]
            
            partners = np.where(aff_mat[i] > 3)[0]
            if len(partners) > 0 and rng.random() < 0.70:
                p = rng.choice(partners)
                p_slot = sku_slots[p]
                d_to_p = dist_mat[p_slot]
                nearby_empty = [s for s in np.where((d_to_p <= 4) & (d_to_p > 0))[0] if slot_to_sku[s] == -1]
                if nearby_empty:
                    s_empty = rng.choice(nearby_empty)
                    empty_idx = empty_slots.index(s_empty)
                else:
                    empty_idx = rng.integers(0, len(empty_slots))
                    s_empty = empty_slots[empty_idx]
            else:
                empty_idx = rng.integers(0, len(empty_slots))
                s_empty = empty_slots[empty_idx]
            
            delta_depot = velocities[i] * (depot_dist[s_empty] - depot_dist[s_i])
            diff_dist = dist_mat[s_empty, sku_slots] - dist_mat[s_i, sku_slots]
            diff_dist[i] = 0.0
            delta_affinity = np.sum(aff_mat[i] * diff_dist)
            
            delta_energy = w_depot * delta_depot + w_affinity * delta_affinity
            
            if delta_energy < 0 or (temp > 1e-4 and rng.random() < np.exp(-delta_energy / temp)):
                sku_slots[i] = s_empty
                slot_to_sku[s_i] = -1
                slot_to_sku[s_empty] = i
                empty_slots[empty_idx] = s_i
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_slots = sku_slots.copy()

        else:
            # Ergodic exploration: random pair swap
            i = rng.integers(0, NUM_SKUS)
            j = rng.integers(0, NUM_SKUS)
            while j == i: j = rng.integers(0, NUM_SKUS)
            
            s_i = sku_slots[i]
            s_j = sku_slots[j]
            delta_depot = (velocities[i] * (depot_dist[s_j] - depot_dist[s_i]) +
                           velocities[j] * (depot_dist[s_i] - depot_dist[s_j]))
            diff_dist_i = dist_mat[s_j, sku_slots] - dist_mat[s_i, sku_slots]
            diff_dist_j = dist_mat[s_i, sku_slots] - dist_mat[s_j, sku_slots]
            delta_aff_k = (aff_mat[i] * diff_dist_i + aff_mat[j] * diff_dist_j)
            delta_aff_k[i] = 0.0
            delta_aff_k[j] = 0.0
            delta_affinity = np.sum(delta_aff_k)
            delta_energy = w_depot * delta_depot + w_affinity * delta_affinity

            if delta_energy < 0 or (temp > 1e-4 and rng.random() < np.exp(-delta_energy / temp)):
                sku_slots[i] = s_j
                sku_slots[j] = s_i
                slot_to_sku[s_i] = j
                slot_to_sku[s_j] = i
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_slots = sku_slots.copy()

        temp *= cooling_rate
        if step % 250 == 0:
            energy_history.append(float(best_energy))

    energy_history.append(float(best_energy))
    best_assignment = {sku: int(best_slots[sku]) for sku in range(NUM_SKUS)}
    return best_assignment, energy_history


def simulate_picker_order_route(
    order: List[int],
    assignment: Dict[int, int],
    env: WarehouseEnvironment
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Simulates a picker traveling from Depot -> SKU_1 -> ... -> SKU_k -> Depot
    Using nearest-neighbor heuristic (standard picking route traversal).
    """
    if not order:
        return 0.0, [DEPOT_COORDS]
    
    unvisited_skus = list(order)
    curr_coord = DEPOT_COORDS
    route_coords = [DEPOT_COORDS]
    total_dist = 0.0
    
    while unvisited_skus:
        best_sku = None
        best_d = float('inf')
        best_coord = None
        
        for sku in unvisited_skus:
            slot_id = assignment[sku]
            coord = env.slot_coords[slot_id]
            d = manhattan_distance(curr_coord, coord)
            if d < best_d:
                best_d = d
                best_sku = sku
                best_coord = coord
        
        total_dist += best_d
        curr_coord = best_coord
        route_coords.append(best_coord)
        unvisited_skus.remove(best_sku)
    
    return_dist = manhattan_distance(curr_coord, DEPOT_COORDS)
    total_dist += return_dist
    route_coords.append(DEPOT_COORDS)
    
    return float(total_dist), route_coords


def evaluate_warehouse_metrics(
    assignment: Dict[int, int],
    env: WarehouseEnvironment,
    w_depot: float = 1.0,
    w_affinity: float = 1.8,
    sample_orders_count: int = 500
) -> Dict[str, Any]:
    """
    Comprehensive KPIs for comparing ABC vs QUBO slotting.
    """
    total_energy, depot_cost, aff_cost = compute_qubo_energy(assignment, env, w_depot, w_affinity)
    
    eval_orders = env.simulated_orders[:sample_orders_count]
    total_picker_dist = 0.0
    order_distances = []
    inter_item_distances = []

    for order in eval_orders:
        dist, _ = simulate_picker_order_route(order, assignment, env)
        total_picker_dist += dist
        order_distances.append(dist)
        
        for i_idx in range(len(order)):
            for j_idx in range(i_idx + 1, len(order)):
                s1 = assignment[order[i_idx]]
                s2 = assignment[order[j_idx]]
                d = env.dist_matrix[s1, s2]
                inter_item_distances.append(d)

    avg_order_distance = float(np.mean(order_distances)) if order_distances else 0.0
    avg_inter_item_dist = float(np.mean(inter_item_distances)) if inter_item_distances else 0.0
    
    class_a_skus = [sku.id for sku in env.skus if sku.abc_class == 'A']
    class_b_skus = [sku.id for sku in env.skus if sku.abc_class == 'B']
    class_c_skus = [sku.id for sku in env.skus if sku.abc_class == 'C']

    avg_depot_a = float(np.mean([env.depot_dist[assignment[s]] for s in class_a_skus]))
    avg_depot_b = float(np.mean([env.depot_dist[assignment[s]] for s in class_b_skus]))
    avg_depot_c = float(np.mean([env.depot_dist[assignment[s]] for s in class_c_skus]))

    cell_meters = 1.5
    total_meters = total_picker_dist * cell_meters
    walking_speed_mps = 1.2
    total_hours = (total_meters / walking_speed_mps) / 3600.0

    return {
        "total_qubo_energy": total_energy,
        "depot_cost": depot_cost,
        "affinity_cost": aff_cost,
        "total_travel_grid_units": total_picker_dist,
        "total_travel_km": round(total_meters / 1000.0, 2),
        "avg_order_travel_units": round(avg_order_distance, 2),
        "avg_order_travel_meters": round(avg_order_distance * cell_meters, 1),
        "avg_inter_item_dist": round(avg_inter_item_dist, 2),
        "total_walking_hours": round(total_hours, 2),
        "avg_depot_dist_class_a": round(avg_depot_a, 2),
        "avg_depot_dist_class_b": round(avg_depot_b, 2),
        "avg_depot_dist_class_c": round(avg_depot_c, 2),
    }


def build_small_subqubo_matrix(
    sku_subset: List[int],
    slot_subset: List[int],
    env: WarehouseEnvironment,
    w_depot: float = 1.0,
    w_affinity: float = 1.8,
    penalty_p: float = 50.0
) -> Tuple[np.ndarray, List[str]]:
    """
    Constructs an explicit algebraic QUBO Q matrix for a small subproblem (k SKUs x m Slots).
    Binary variable: x_{i, s} in {0, 1}
    Cost = x^T Q x
    """
    k = len(sku_subset)
    m = len(slot_subset)
    num_vars = k * m
    
    var_labels = []
    for i_idx, sku_id in enumerate(sku_subset):
        for s_idx, slot_id in enumerate(slot_subset):
            coord = env.slot_coords[slot_id]
            var_labels.append(f"x({env.skus[sku_id].name}, [{coord[0]},{coord[1]}])")
    
    Q = np.zeros((num_vars, num_vars), dtype=np.float64)
    
    def get_var(i_i, s_i):
        return i_i * m + s_i

    # 1. Linear Depot travel cost on the diagonal (x_u^2 = x_u)
    for i_idx, sku_id in enumerate(sku_subset):
        v = env.skus[sku_id].velocity
        for s_idx, slot_id in enumerate(slot_subset):
            u = get_var(i_idx, s_idx)
            d = env.depot_dist[slot_id]
            Q[u, u] += w_depot * v * d

    # 2. Quadratic Affinity interaction between SKUs
    for i1 in range(k):
        for i2 in range(i1 + 1, k):
            sku1 = sku_subset[i1]
            sku2 = sku_subset[i2]
            aff = env.affinity_matrix[sku1, sku2]
            if aff > 0:
                for s1 in range(m):
                    for s2 in range(m):
                        if s1 != s2:
                            u1 = get_var(i1, s1)
                            u2 = get_var(i2, s2)
                            slot1 = slot_subset[s1]
                            slot2 = slot_subset[s2]
                            d = env.dist_matrix[slot1, slot2]
                            quad_cost = w_affinity * aff * d
                            Q[u1, u2] += quad_cost / 2.0
                            Q[u2, u1] += quad_cost / 2.0

    # 3. Penalty Constraint: Each SKU assigned to exactly one slot: P * (sum_s x_{i,s} - 1)^2
    for i_idx in range(k):
        for s_idx in range(m):
            u = get_var(i_idx, s_idx)
            Q[u, u] += -penalty_p
        
        for s1 in range(m):
            for s2 in range(s1 + 1, m):
                u1 = get_var(i_idx, s1)
                u2 = get_var(i_idx, s2)
                Q[u1, u2] += penalty_p
                Q[u2, u1] += penalty_p

    # 4. Penalty Constraint: At most one SKU per slot: P * sum_{i < i'} x_{i,s} x_{i',s}
    for s_idx in range(m):
        for i1 in range(k):
            for i2 in range(i1 + 1, k):
                u1 = get_var(i1, s_idx)
                u2 = get_var(i2, s_idx)
                Q[u1, u2] += penalty_p / 2.0
                Q[u2, u1] += penalty_p / 2.0

    return Q, var_labels
