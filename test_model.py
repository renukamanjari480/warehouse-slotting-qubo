"""
Quick validation test for warehouse_model.py
"""
import time
from warehouse_model import (
    generate_warehouse_data,
    solve_abc_slotting,
    solve_qubo_slotting_simulated_annealing,
    evaluate_warehouse_metrics,
    build_small_subqubo_matrix,
    simulate_picker_order_route,
)

def main():
    print("Testing warehouse data generation...")
    t0 = time.time()
    env = generate_warehouse_data(seed=42)
    print(f"Generated {len(env.skus)} SKUs, {len(env.slot_coords)} slots in {time.time()-t0:.2f}s")
    
    print("Testing ABC slotting...")
    t0 = time.time()
    abc_res = solve_abc_slotting(env)
    print(f"ABC slotted {len(abc_res)} items in {time.time()-t0:.4f}s")
    
    abc_metrics = evaluate_warehouse_metrics(abc_res, env, sample_orders_count=100)
    print(f"ABC Total Travel: {abc_metrics['total_travel_km']} km, Avg Order: {abc_metrics['avg_order_travel_meters']} m")
    
    print("Testing QUBO simulated annealing...")
    t0 = time.time()
    qubo_res, hist = solve_qubo_slotting_simulated_annealing(env, max_steps=5000, seed=42)
    print(f"QUBO solved 5000 steps in {time.time()-t0:.2f}s, start energy: {hist[0]:.1f}, end energy: {hist[-1]:.1f}")
    
    qubo_metrics = evaluate_warehouse_metrics(qubo_res, env, sample_orders_count=100)
    print(f"QUBO Total Travel: {qubo_metrics['total_travel_km']} km, Avg Order: {qubo_metrics['avg_order_travel_meters']} m")
    
    improvement = (abc_metrics['total_travel_grid_units'] - qubo_metrics['total_travel_grid_units']) / abc_metrics['total_travel_grid_units'] * 100
    print(f"Travel distance improvement: {improvement:.2f}%")
    
    print("Testing subqubo matrix...")
    Q, labels = build_small_subqubo_matrix([0, 1, 2], [0, 1, 2, 3], env)
    print(f"Sub-QUBO matrix shape: {Q.shape}, variables: {len(labels)}")
    print("All tests passed successfully!")

if __name__ == "__main__":
    main()
