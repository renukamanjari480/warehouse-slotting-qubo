"""
app.py - Quantum Warehouse Slotting Optimizer: Before (ABC) vs After (QUBO)
A Streamlit web application featuring a 20x20 warehouse grid (400 slots) with 100 SKUs,
comparing classical ABC slotting heuristics against QUBO (Quadratic Unconstrained Binary Optimization).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from warehouse_model import (
    GRID_SIZE,
    TOTAL_SLOTS,
    NUM_SKUS,
    DEPOT_COORDS,
    CATEGORIES,
    generate_warehouse_data,
    solve_abc_slotting,
    solve_qubo_slotting_simulated_annealing,
    simulate_picker_order_route,
    evaluate_warehouse_metrics,
    build_small_subqubo_matrix,
    manhattan_distance,
)

# Set page configuration
st.set_page_config(
    page_title="Quantum Warehouse Slotting: ABC vs QUBO",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern executive aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E88E5 0%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #718096;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-badge-green {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 9999px;
        margin-top: 4px;
    }
    .kpi-badge-neutral {
        display: inline-block;
        background: #e2e8f0;
        color: #334155;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 9999px;
        margin-top: 4px;
    }
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# --- State Management & Cached Solvers ---
@st.cache_data(show_spinner=False)
def get_warehouse_env(seed: int):
    return generate_warehouse_data(seed=seed, num_orders=1200)

@st.cache_data(show_spinner=False)
def get_abc_solution(_env):
    return solve_abc_slotting(_env)

@st.cache_data(show_spinner=False)
def get_qubo_solution(_env, w_depot: float, w_affinity: float, max_steps: int, seed: int):
    return solve_qubo_slotting_simulated_annealing(
        _env,
        w_depot=w_depot,
        w_affinity=w_affinity,
        max_steps=max_steps,
        seed=seed
    )


# --- Helper: Plotly Warehouse 20x20 Grid ---
def render_warehouse_grid(
    env,
    assignment: dict,
    title: str,
    selected_sku: int = None,
    order_route: list = None,
    color_by: str = "ABC Class"
):
    """
    Renders an interactive 20x20 warehouse grid using Plotly.
    - Depot marked at (0, 0)
    - 400 slots (100 active SKUs, 300 empty slots)
    - Color coding: Class A (Red), Class B (Yellow), Class C (Blue), Empty (Light Gray)
    - Optional interactive overlay: top affinity links or picker order travel route
    """
    # Create empty grid matrix
    grid_x = []
    grid_y = []
    slot_ids = []
    sku_labels = []
    sku_classes = []
    sku_cats = []
    sku_vels = []
    colors = []
    hover_texts = []
    symbols = []
    marker_sizes = []

    # Invert assignment: slot_id -> sku_id
    slot_to_sku = {s: -1 for s in range(TOTAL_SLOTS)}
    for s_id, slot in assignment.items():
        slot_to_sku[slot] = s_id

    # Color map for ABC classes
    abc_colors = {
        'A': '#EF4444',  # Red / Crimson (High Velocity)
        'B': '#F59E0B',  # Amber / Orange (Medium Velocity)
        'C': '#3B82F6',  # Bright Blue (Low Velocity)
        'Empty': '#E2E8F0'  # Soft Slate
    }

    category_colors = {c["name"]: c["color"] for c in CATEGORIES}

    for s_idx in range(TOTAL_SLOTS):
        coord = env.slot_coords[s_idx]
        gx, gy = coord[0], coord[1]
        grid_x.append(gx)
        grid_y.append(gy)
        slot_ids.append(s_idx)

        sku_id = slot_to_sku[s_idx]
        if sku_id != -1:
            sku = env.skus[sku_id]
            sku_labels.append(sku.name)
            sku_classes.append(sku.abc_class)
            sku_cats.append(sku.category)
            sku_vels.append(sku.velocity)
            
            if color_by == "ABC Class":
                c = abc_colors[sku.abc_class]
            elif color_by == "Category":
                c = category_colors.get(sku.category, '#94A3B8')
            else:  # Velocity heatmap
                c = sku.velocity
            colors.append(c)

            # Highlight selected SKU
            if selected_sku is not None and sku_id == selected_sku:
                symbols.append('diamond')
                marker_sizes.append(18)
            else:
                symbols.append('square')
                marker_sizes.append(13)

            d_depot = env.depot_dist[s_idx]
            ht = (
                f"<b>{sku.name}</b> (Class {sku.abc_class})<br>"
                f"Location: Bay ({gx}, {gy}) [Slot #{s_idx}]<br>"
                f"Category: {sku.category}<br>"
                f"Daily Velocity: {sku.velocity:.0f} picks<br>"
                f"Dist to Depot: {d_depot:.0f}m"
            )
            hover_texts.append(ht)
        else:
            sku_labels.append("Empty")
            sku_classes.append("Empty")
            sku_cats.append("Unassigned")
            sku_vels.append(0)
            colors.append(abc_colors['Empty'])
            symbols.append('square')
            marker_sizes.append(8)
            d_depot = env.depot_dist[s_idx]
            hover_texts.append(f"<b>Empty Slot #{s_idx}</b><br>Coords: ({gx}, {gy})<br>Dist to Depot: {d_depot:.0f}m")

    fig = go.Figure()

    # Base grid scatter
    fig.add_trace(go.Scatter(
        x=grid_x,
        y=grid_y,
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=colors,
            symbol=symbols,
            line=dict(width=1, color='#64748b'),
            opacity=0.9
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Storage Slots'
    ))

    # Mark I/O Depot at (0, 0)
    fig.add_trace(go.Scatter(
        x=[DEPOT_COORDS[0]],
        y=[DEPOT_COORDS[1]],
        mode='markers+text',
        marker=dict(size=24, color='#10B981', symbol='star', line=dict(width=2, color='#047857')),
        text=["DEPOT (I/O)"],
        textposition="top right",
        textfont=dict(color="#047857", size=11, family="Arial Black"),
        hovertext="<b>DEPOT / LOADING DOCK (0, 0)</b><br>All pickers depart and return here.",
        hoverinfo='text',
        name='Depot / Dock'
    ))

    # Highlight affinity lines if a SKU is selected
    if selected_sku is not None:
        sel_slot = assignment[selected_sku]
        sel_coord = env.slot_coords[sel_slot]
        # Find top 4 affinity partners
        aff_row = env.affinity_matrix[selected_sku].copy()
        aff_row[selected_sku] = 0
        top_partners = np.argsort(-aff_row)[:4]

        for p_sku in top_partners:
            p_aff = aff_row[p_sku]
            if p_aff > 0:
                p_slot = assignment[p_sku]
                p_coord = env.slot_coords[p_slot]
                p_name = env.skus[p_sku].name
                fig.add_trace(go.Scatter(
                    x=[sel_coord[0], p_coord[0]],
                    y=[sel_coord[1], p_coord[1]],
                    mode='lines+markers',
                    line=dict(color='#8B5CF6', width=2.5, dash='dot'),
                    marker=dict(size=10, color='#8B5CF6', symbol='circle'),
                    hovertext=f"Affinity Link: {env.skus[selected_sku].name} & {p_name}<br>Co-picks: {p_aff:.0f} orders<br>Walking separation: {manhattan_distance(sel_coord, p_coord):.0f}m",
                    hoverinfo='text',
                    name=f"Link to {p_name}",
                    showlegend=False
                ))

    # Overlay simulated order pick route
    if order_route and len(order_route) > 1:
        rx = [c[0] for c in order_route]
        ry = [c[1] for c in order_route]
        fig.add_trace(go.Scatter(
            x=rx,
            y=ry,
            mode='lines+markers+text',
            line=dict(color='#EC4899', width=3),
            marker=dict(size=9, color='#BE185D', symbol='circle'),
            text=[f"{i}" if i > 0 and i < len(rx)-1 else "" for i in range(len(rx))],
            textposition="top center",
            textfont=dict(color="#BE185D", size=10, family="Arial Black"),
            hovertext=[f"Step {i}: ({c[0]}, {c[1]})" for i, c in enumerate(order_route)],
            hoverinfo='text',
            name='Order Pick Path'
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1e293b", family="sans-serif")),
        xaxis=dict(
            title="Warehouse Bay X (0 to 19)",
            range=[-1, 20],
            dtick=2,
            gridcolor="#f1f5f9",
            zeroline=False,
            showgrid=True
        ),
        yaxis=dict(
            title="Warehouse Bay Y (0 to 19)",
            range=[-1, 20],
            dtick=2,
            gridcolor="#f1f5f9",
            zeroline=False,
            showgrid=True,
            scaleanchor="x",
            scaleratio=1
        ),
        height=580,
        margin=dict(l=40, r=40, t=50, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        )
    )
    return fig


# --- Main Application Header ---
st.markdown('<div class="main-title">Warehouse Slotting Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    '<b>Before vs After:</b> Comparing Classical Pareto <b>ABC Velocity Slotting</b> against <b>QUBO (Quantum / Quadratic Unconstrained Binary Optimization)</b> '
    'on a <b>20 × 20 Grid (400 Slots)</b> with <b>100 SKUs</b>.'
    '</div>',
    unsafe_allow_html=True
)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Warehouse Configuration")
    sim_seed = st.number_input("Random Seed (Data Generator)", min_value=1, max_value=9999, value=42, step=1)
    eval_orders_count = st.slider("Evaluation Multi-Item Orders", min_value=100, max_value=1200, value=500, step=50)

    st.markdown("---")
    st.header("⚛️ QUBO Solver Parameters")
    st.caption("Balance direct Depot access vs SKU-to-SKU Affinity clustering.")
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        w_depot = st.slider("Depot Weight (w₁)", 0.2, 2.5, 1.0, 0.1, help="Linear cost penalty for distance from I/O Depot")
    with col_w2:
        w_affinity = st.slider("Affinity Weight (w₂)", 0.2, 3.5, 1.8, 0.1, help="Quadratic penalty for distance between frequently co-ordered SKUs")

    anneal_steps = st.select_slider(
        "Simulated Annealing Steps",
        options=[5000, 10000, 15000, 20000, 30000],
        value=15000,
        help="Cooling schedule iterations for quantum-inspired optimization"
    )
    
    rerun_opt = st.button("🚀 Re-run QUBO Optimization", use_container_width=True, type="primary")

    st.markdown("---")
    st.header("🔍 Interactive Inspector")
    color_scheme = st.radio("Grid Color Encoding:", ["ABC Class", "Category"], horizontal=True)
    
    # Load environment
    env = get_warehouse_env(sim_seed)
    
    sku_options = [-1] + list(range(NUM_SKUS))
    selected_sku_idx = st.selectbox(
        "Highlight SKU & Affinity Partners:",
        options=sku_options,
        format_func=lambda x: "None (Show All)" if x == -1 else f"{env.skus[x].name} - {env.skus[x].category} ({env.skus[x].abc_class})",
        index=6  # default highlight SKU-006
    )
    selected_sku = None if selected_sku_idx == -1 else selected_sku_idx

    st.markdown("---")
    st.header("🛒 Order Route Simulation")
    simulate_order = st.checkbox("Simulate Pick Route on Grids", value=True)
    order_idx = st.slider("Select Order # from History:", 0, 99, 12)


# --- Load & Compute Models ---
with st.spinner("Generating warehouse topology and solving slotting models..."):
    # 1. ABC Slotting Solution (Before)
    abc_assignment = get_abc_solution(env)
    abc_metrics = evaluate_warehouse_metrics(abc_assignment, env, w_depot, w_affinity, eval_orders_count)

    # 2. QUBO Slotting Solution (After)
    qubo_assignment, energy_history = get_qubo_solution(
        env, w_depot=w_depot, w_affinity=w_affinity, max_steps=anneal_steps, seed=sim_seed
    )
    qubo_metrics = evaluate_warehouse_metrics(qubo_assignment, env, w_depot, w_affinity, eval_orders_count)

# Compute simulated route if enabled
active_order = env.simulated_orders[order_idx] if simulate_order else []
abc_order_dist, abc_route = simulate_picker_order_route(active_order, abc_assignment, env) if simulate_order else (0, [])
qubo_order_dist, qubo_route = simulate_picker_order_route(active_order, qubo_assignment, env) if simulate_order else (0, [])


# --- Top Level Comparative KPI Scorecard ---
dist_saved_pct = ((abc_metrics["total_travel_grid_units"] - qubo_metrics["total_travel_grid_units"]) / abc_metrics["total_travel_grid_units"]) * 100
order_saved_pct = ((abc_metrics["avg_order_travel_units"] - qubo_metrics["avg_order_travel_units"]) / abc_metrics["avg_order_travel_units"]) * 100
inter_saved_pct = ((abc_metrics["avg_inter_item_dist"] - qubo_metrics["avg_inter_item_dist"]) / abc_metrics["avg_inter_item_dist"]) * 100
energy_saved_pct = ((abc_metrics["total_qubo_energy"] - qubo_metrics["total_qubo_energy"]) / abc_metrics["total_qubo_energy"]) * 100

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Picker Travel (500 Orders)</div>
        <div class="kpi-value">{qubo_metrics['total_travel_km']} km</div>
        <span class="kpi-badge-green">⬇ {dist_saved_pct:.1f}% vs ABC ({abc_metrics['total_travel_km']} km)</span>
    </div>
    """, unsafe_allow_html=True)

with kpi_c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Avg Walking Distance / Order</div>
        <div class="kpi-value">{qubo_metrics['avg_order_travel_meters']} m</div>
        <span class="kpi-badge-green">⬇ {order_saved_pct:.1f}% vs ABC ({abc_metrics['avg_order_travel_meters']} m)</span>
    </div>
    """, unsafe_allow_html=True)

with kpi_c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Avg Inter-Item Distance</div>
        <div class="kpi-value">{qubo_metrics['avg_inter_item_dist']} bays</div>
        <span class="kpi-badge-green">⬇ {inter_saved_pct:.1f}% Co-location Gain</span>
    </div>
    """, unsafe_allow_html=True)

with kpi_c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Hamiltonian Objective Cost</div>
        <div class="kpi-value">{qubo_metrics['total_qubo_energy']/1e3:.1f}k</div>
        <span class="kpi-badge-green">⬇ {energy_saved_pct:.1f}% Lower Cost</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# --- Core Two Tabs requested: Left Tab = ABC Slotting, Right Tab = QUBO Slotting ---
tab_abc, tab_qubo, tab_comparison, tab_quantum_math = st.tabs([
    "📦 Left Tab: ABC Slotting (Before)",
    "⚛️ Right Tab: QUBO Optimized Slotting (After)",
    "📊 Side-by-Side Path Comparison & Analytics",
    "🔬 QUBO Mathematical Formulation & Q Matrix"
])


# ==============================================================================
# TAB 1: ABC SLOTTING (BEFORE WAREHOUSE)
# ==============================================================================
with tab_abc:
    st.markdown("### 📦 Before Warehouse: Classical Pareto ABC Slotting")
    st.markdown("""
    <div class="info-box">
        <b>How ABC Heuristic Works:</b> Slotting is allocated strictly on SKU <b>velocity</b> (popularity). 
        <b>Class A</b> items (top 20 SKUs = 70% of picks) are packed into the slots closest to the Depot at (0, 0). 
        <b>Class B</b> (30 SKUs) occupy the mid-tier ring, and <b>Class C</b> (50 SKUs) are pushed to the outer boundary.
        <br><br>
        ⚠️ <b>The Major Flaw:</b> ABC slotting is <i>affinity-blind</i>. If a Class A item (e.g. SKU-006 smartphone) is frequently purchased with a Class C companion (e.g. SKU-092 protective cable), 
        the picker must sprint all the way from the depot to the outer perimeter!
    </div>
    """, unsafe_allow_html=True)

    col_abc_grid, col_abc_details = st.columns([2.3, 1.2])

    with col_abc_grid:
        fig_abc = render_warehouse_grid(
            env=env,
            assignment=abc_assignment,
            title="Before: ABC Slotting Heuristic (Strict Distance-from-Depot Concentric Rings)",
            selected_sku=selected_sku,
            order_route=abc_route if simulate_order else None,
            color_by=color_scheme
        )
        st.plotly_chart(fig_abc, use_container_width=True)

    with col_abc_details:
        st.subheader("📋 ABC Layout Metrics")
        st.metric("Total Travel (500 Orders)", f"{abc_metrics['total_travel_km']} km")
        st.metric("Avg Walking Time", f"{abc_metrics['total_walking_hours']} hrs")
        st.metric("Avg Order Travel", f"{abc_metrics['avg_order_travel_meters']} meters")
        st.metric("Avg Inter-Item Distance", f"{abc_metrics['avg_inter_item_dist']} bays")

        st.markdown("---")
        st.markdown("**Mean Distance to Depot by Tier:**")
        st.write(f"• **Class A (20 SKUs):** {abc_metrics['avg_depot_dist_class_a']:.1f} bays")
        st.write(f"• **Class B (30 SKUs):** {abc_metrics['avg_depot_dist_class_b']:.1f} bays")
        st.write(f"• **Class C (50 SKUs):** {abc_metrics['avg_depot_dist_class_c']:.1f} bays")

        if simulate_order and active_order:
            st.markdown("---")
            st.markdown(f"**Current Pick Order #{order_idx}:**")
            order_sku_names = [env.skus[i].name for i in active_order]
            st.write(f"Items ({len(active_order)}): `{' + '.join(order_sku_names)}`")
            st.markdown(f"**ABC Route Distance:** <span style='color:#EF4444; font-weight:bold;'>{abc_order_dist * 1.5:.1f} meters</span>", unsafe_allow_html=True)


# ==============================================================================
# TAB 2: QUBO OPTIMIZED SLOTTING (AFTER WAREHOUSE)
# ==============================================================================
with tab_qubo:
    st.markdown("### ⚛️ After Warehouse: QUBO-Optimized Slotting")
    st.markdown("""
    <div class="info-box" style="border-left-color: #7C4DFF; background: #f5f3ff; color: #4c1d95;">
        <b>How QUBO Optimization Works:</b> Modeled as a Quadratic Assignment Problem (QAP) solved via Quantum-Inspired Simulated Annealing.
        It jointly optimizes two competing physical objectives:
        <br>
        1. <b>Linear Term (Velocity):</b> Keep high-frequency SKUs within reasonable reach of the I/O Depot.
        <br>
        2. <b>Quadratic Coupling Term (Affinity Matrix <i>A<sub>ij</sub></i>):</b> Pull strongly correlated product bundles (e.g. Electronics, Tools, Personal Care) into tightly co-located warehouse clusters.
        <br><br>
        ✨ <b>The Breakthrough:</b> Frequently co-picked Class B and C SKUs are now strategically slotted adjacent to their Class A companion items, drastically cutting zigzagging!
    </div>
    """, unsafe_allow_html=True)

    col_qubo_grid, col_qubo_details = st.columns([2.3, 1.2])

    with col_qubo_grid:
        fig_qubo = render_warehouse_grid(
            env=env,
            assignment=qubo_assignment,
            title="After: QUBO Optimized Slotting (Co-location Clusters & Strategic Proximity)",
            selected_sku=selected_sku,
            order_route=qubo_route if simulate_order else None,
            color_by=color_scheme
        )
        st.plotly_chart(fig_qubo, use_container_width=True)

    with col_qubo_details:
        st.subheader("🚀 QUBO Performance Gains")
        st.metric(
            "Total Travel (500 Orders)",
            f"{qubo_metrics['total_travel_km']} km",
            delta=f"-{dist_saved_pct:.1f}%",
            delta_color="normal"
        )
        st.metric(
            "Avg Walking Time Saved",
            f"{qubo_metrics['total_walking_hours']} hrs",
            delta=f"-{(abc_metrics['total_walking_hours'] - qubo_metrics['total_walking_hours']):.1f} hrs",
            delta_color="normal"
        )
        st.metric(
            "Avg Order Travel",
            f"{qubo_metrics['avg_order_travel_meters']} meters",
            delta=f"-{order_saved_pct:.1f}%",
            delta_color="normal"
        )
        st.metric(
            "Avg Inter-Item Distance",
            f"{qubo_metrics['avg_inter_item_dist']} bays",
            delta=f"-{inter_saved_pct:.1f}%",
            delta_color="normal"
        )

        st.markdown("---")
        st.markdown("**Mean Distance to Depot by Tier:**")
        st.write(f"• **Class A:** {qubo_metrics['avg_depot_dist_class_a']:.1f} bays")
        st.write(f"• **Class B:** {qubo_metrics['avg_depot_dist_class_b']:.1f} bays")
        st.write(f"• **Class C:** {qubo_metrics['avg_depot_dist_class_c']:.1f} bays")

        if simulate_order and active_order:
            st.markdown("---")
            st.markdown(f"**Current Pick Order #{order_idx}:**")
            st.markdown(f"**QUBO Route Distance:** <span style='color:#10B981; font-weight:bold;'>{qubo_order_dist * 1.5:.1f} meters</span> (Saved: <b>{(abc_order_dist - qubo_order_dist)*1.5:.1f} m</b>)", unsafe_allow_html=True)


# ==============================================================================
# TAB 3: SIDE-BY-SIDE PATH COMPARISON & ANALYTICS
# ==============================================================================
with tab_comparison:
    st.markdown("### 📊 Side-by-Side Visual Proof: Order Route Simulation")
    st.caption("Inspect the exact picking journey for both warehouse configurations simultaneously.")

    if simulate_order and active_order:
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown(f"#### 🔴 Before (ABC Heuristic) — Route: {abc_order_dist * 1.5:.1f} meters")
            st.plotly_chart(
                render_warehouse_grid(env, abc_assignment, "ABC: Dispersed Route Across Warehouse", selected_sku=None, order_route=abc_route, color_by="ABC Class"),
                use_container_width=True
            )
        with comp_col2:
            st.markdown(f"#### 🟢 After (QUBO Optimization) — Route: {qubo_order_dist * 1.5:.1f} meters")
            st.plotly_chart(
                render_warehouse_grid(env, qubo_assignment, f"QUBO: Compact Co-located Pick Route (Saved {((abc_order_dist - qubo_order_dist)/abc_order_dist)*100:.1f}%)", selected_sku=None, order_route=qubo_route, color_by="ABC Class"),
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("### 📈 Optimization Analytics & Convergence History")
    
    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        # Energy Descent Plot
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=[i * 250 for i in range(len(energy_history))],
            y=energy_history,
            mode='lines',
            line=dict(color='#7C4DFF', width=2.5),
            name='QUBO Hamiltonian Energy'
        ))
        fig_conv.update_layout(
            title="Simulated Annealing Energy Minimization Curve",
            xaxis_title="Annealing Iteration Step",
            yaxis_title="Objective Energy (Hamiltonian Cost)",
            height=360,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    with chart_c2:
        # Cost Breakdown Bar Chart
        cost_df = pd.DataFrame({
            "Metric": ["Depot Travel Cost", "Affinity Interaction Cost", "Total Hamiltonian Energy"],
            "ABC (Before)": [abc_metrics["depot_cost"], abc_metrics["affinity_cost"], abc_metrics["total_qubo_energy"]],
            "QUBO (After)": [qubo_metrics["depot_cost"], qubo_metrics["affinity_cost"], qubo_metrics["total_qubo_energy"]]
        })
        
        fig_bar = go.Figure(data=[
            go.Bar(name='Before: ABC', x=cost_df['Metric'], y=cost_df['Before: ABC'] if 'Before: ABC' in cost_df else cost_df['ABC (Before)'], marker_color='#EF4444'),
            go.Bar(name='After: QUBO', x=cost_df['Metric'], y=cost_df['After: QUBO'] if 'After: QUBO' in cost_df else cost_df['QUBO (After)'], marker_color='#10B981')
        ])
        fig_bar.update_layout(
            title="Depot Distance vs Affinity Clustering Cost Trade-off",
            barmode='group',
            height=360,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Full SKU Slotting Table
    st.markdown("### 📋 100 SKUs Allocation Catalog")
    table_rows = []
    for s in env.skus:
        abc_s = abc_assignment[s.id]
        qubo_s = qubo_assignment[s.id]
        abc_c = env.slot_coords[abc_s]
        qubo_c = env.slot_coords[qubo_s]
        abc_depot = env.depot_dist[abc_s]
        qubo_depot = env.depot_dist[qubo_s]
        
        table_rows.append({
            "SKU ID": s.name,
            "Category": s.category,
            "Class": s.abc_class,
            "Velocity (Picks/Day)": s.velocity,
            "ABC Slot (x, y)": f"({abc_c[0]}, {abc_c[1]}) [Slot #{abc_s}]",
            "ABC Depot Dist": f"{abc_depot:.0f}m",
            "QUBO Slot (x, y)": f"({qubo_c[0]}, {qubo_c[1]}) [Slot #{qubo_s}]",
            "QUBO Depot Dist": f"{qubo_depot:.0f}m",
            "Delta Depot Dist": f"{qubo_depot - abc_depot:+.0f}m"
        })
    df_catalog = pd.DataFrame(table_rows)
    st.dataframe(df_catalog, use_container_width=True, height=300)

    # Download CSV
    csv_data = df_catalog.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Warehouse Slotting Plan (CSV)",
        data=csv_data,
        file_name="qubo_warehouse_slotting_plan.csv",
        mime="text/csv"
    )


# ==============================================================================
# TAB 4: QUBO MATHEMATICAL FORMULATION & SUB-QUBO MATRIX INSPECTOR
# ==============================================================================
with tab_quantum_math:
    st.markdown("### 🔬 Mathematical Formulation: From Warehouse to QUBO")
    st.markdown(r"""
    Warehouse slotting is a classic NP-hard **Quadratic Assignment Problem (QAP)**. To execute this on quantum annealers 
    (e.g., D-Wave Advantage) or quantum-inspired digital annealers, we formulate it as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem:
    
    $$\min_{x} \quad H(x) = x^T Q x$$
    
    Where binary decision variable $x_{i, s} \in \{0, 1\}$ denotes whether **SKU $i$** is assigned to **Slot $s$**.
    """)

    st.markdown(r"""
    #### The Objective Hamiltonian:
    $$H(x) = \underbrace{w_1 \sum_{i=1}^N \sum_{s=1}^M v_i \, d(s, \text{depot}) \, x_{i, s}}_{\text{Linear: Velocity to Loading Dock}} + \underbrace{w_2 \sum_{i < j}^N \sum_{s \neq s'}^M A_{ij} \, d(s, s') \, x_{i, s} x_{j, s'}}_{\text{Quadratic: Product Affinity \& Co-location}} + \underbrace{P \sum_{i=1}^N \left(\sum_{s=1}^M x_{i, s} - 1\right)^2}_{\text{Penalty: Exactly 1 slot per SKU}} + \underbrace{P \sum_{s=1}^M \sum_{i < j} x_{i, s} x_{j, s}}_{\text{Penalty: At most 1 SKU per slot}}$$
    
    * $v_i$: Daily picking demand/velocity of SKU $i$.
    * $d(s, \text{depot})$: Manhattan distance from slot $s$ to the I/O dock.
    * $A_{ij}$: Co-occurrence frequency of SKUs $i$ and $j$ in multi-item order histories.
    * $d(s, s')$: Inter-slot travel distance between slot $s$ and slot $s'$.
    * $P$: Penalty multiplier enforcing physical 1-to-1 matching constraints without violating binary relaxation.
    """)

    st.markdown("---")
    st.markdown("#### 🔍 Interactive Sub-QUBO Matrix Heatmap")
    st.caption("Select a mini-subset of SKUs and Slots below to inspect the actual numeric $Q$ matrix coupling terms!")

    sub_skus_selected = st.multiselect(
        "Choose 3 to 5 SKUs for Sub-QUBO Matrix:",
        options=list(range(NUM_SKUS)),
        default=[0, 1, 5, 22],
        format_func=lambda x: f"{env.skus[x].name} (Class {env.skus[x].abc_class})"
    )
    
    sub_slots_selected = st.multiselect(
        "Choose 4 to 6 Candidate Slots:",
        options=list(range(TOTAL_SLOTS)),
        default=[0, 1, 20, 21, 40],
        format_func=lambda x: f"Slot #{x} at ({env.slot_coords[x][0]}, {env.slot_coords[x][1]})"
    )

    if len(sub_skus_selected) >= 2 and len(sub_slots_selected) >= 2:
        Q_matrix, var_names = build_small_subqubo_matrix(
            sub_skus_selected, sub_slots_selected, env, w_depot=w_depot, w_affinity=w_affinity, penalty_p=200.0
        )
        
        fig_q = go.Figure(data=go.Heatmap(
            z=Q_matrix,
            x=var_names,
            y=var_names,
            colorscale='Viridis',
            hoverongaps=False
        ))
        fig_q.update_layout(
            title=f"Sub-QUBO Coupling Matrix Q ({len(var_names)} × {len(var_names)} Variables)",
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
            height=520,
            margin=dict(l=80, r=40, t=50, b=120)
        )
        st.plotly_chart(fig_q, use_container_width=True)
        st.info("💡 **Diagonal entries $Q_{u,u}$** represent linear velocity costs minus penalty relaxation. **Off-diagonal entries $Q_{u,v}$** represent quadratic affinity interactions and mutual exclusion penalties.")
    else:
        st.warning("Please select at least 2 SKUs and 2 Slots to view the Sub-QUBO Matrix.")
