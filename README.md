# Quantum Warehouse Slotting Optimizer: Before (ABC) vs After (QUBO)

An interactive Streamlit application comparing classical **ABC Velocity Slotting** with **QUBO (Quadratic Unconstrained Binary Optimization)** on a **20 × 20 grid (400 storage slots)** with **100 SKUs**.

---

## 🚀 Key Highlights & Capabilities

- **20 × 20 Warehouse Topology (400 Slots)**:
  - 100 Active Stock Keeping Units (SKUs)
  - 300 Expansion / Empty bays
  - Input/Output Loading Dock (Depot) located at bay `(0, 0)`

- **Left Tab: 📦 Before Warehouse (ABC Slotting Heuristic)**:
  - Classical Pareto slotting: Class A (top 20 SKUs = ~70% volume), Class B (30 SKUs = ~20%), Class C (50 SKUs = ~10%).
  - Sorted strictly by orthogonal Manhattan distance to the I/O Depot.
  - Highlights the core flaw: **Affinity blindness** (frequently co-ordered items end up split across opposite ends of the warehouse).

- **Right Tab: ⚛️ After Warehouse (QUBO Optimized Slotting)**:
  - Formulated as a **Quadratic Assignment Problem (QAP)** in QUBO form:
    $$\min_{x} \quad H(x) = w_1 \sum_{i, s} v_i d(s, \text{depot}) x_{i, s} + w_2 \sum_{i < j, s \neq s'} A_{ij} d(s, s') x_{i, s} x_{j, s'} + \text{Penalties}$$
  - Solved using fast Quantum-Inspired Simulated Annealing with $O(\text{deg})$ state transition delta evaluations.
  - Generates compact co-location clusters for complementary SKUs near the depot.

- **Interactive Features**:
  - **SKU & Bundle Inspector**: Select any SKU to highlight its location and inspect the physical connection lines to its top co-ordered companion SKUs.
  - **Live Order Route Simulation**: Simulates a picker route from Depot $\to$ items in order $\to$ Depot, showing the exact walking path comparison.
  - **Sub-QUBO Matrix Heatmap**: Live inspection of the algebraic $Q$ matrix ($x^T Q x$) for quantum annealers (such as D-Wave Advantage).
  - **KPI Scorecards & CSV Export**: Real-time metrics on kilometers walked, picker hours, inter-item separation, and downloadable WMS slotting plan.

---

## 🏃 Quick Start

Run the app using Streamlit:
```bash
streamlit run app.py
```
Or with explicit port:
```bash
streamlit run app.py --server.port 8501
```
