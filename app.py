"""
Streamlit + Folium Delivery Route Optimizer
Features:
- Upload CSV with delivery points (id, lat, lon, [address])
- Cluster orders into k zones (KMeans)
- Compute routes per driver using Nearest Neighbor + 2-opt improvement
- "Quantum" mode runs a stronger 2-opt + random-restart heuristic (quantum-inspired)
- Show map with markers and polylines via Folium (embedded using streamlit_folium)
- Add traffic-jam points to increase nearby distances (simulate rerouting)
- Environmental impact: distance, fuel saved estimate, CO2 saved estimate
- Download optimized routes as CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from io import StringIO, BytesIO
import base64
import math
import time
import random

st.set_page_config(page_title="Quantum Delivery Command Center (Demo)", layout="wide")

# -------------------------
# Utility functions
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    # returns distance in km between two lat/lon
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2*R*math.asin(math.sqrt(a))

def pairwise_distance_matrix(points, traffic_jams=None, jam_influence_km=2.0, jam_penalty_factor=2.0):
    # points: list of (lat, lon)
    n = len(points)
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j: mat[i][j]=0.0
            else:
                d = haversine(points[i][0], points[i][1], points[j][0], points[j][1])
                # If traffic_jams present, increase distance if segment passes near jam (approx)
                if traffic_jams:
                    for jam in traffic_jams:
                        # if either endpoint within influence, increase cost
                        if haversine(points[i][0], points[i][1], jam[0], jam[1]) <= jam_influence_km or \
                           haversine(points[j][0], points[j][1], jam[0], jam[1]) <= jam_influence_km:
                            d *= jam_penalty_factor
                            break
                mat[i][j] = d
    return mat

def route_length(route, distmat):
    s = 0.0
    for i in range(len(route)-1):
        s += distmat[route[i]][route[i+1]]
    return s

def nearest_neighbor_tour(start_idx, indices, distmat):
    # start at start_idx, visit all indices (list) returning a route (list of indices)
    unvisited = set(indices)
    route = [start_idx]
    current = start_idx
    if current in unvisited:
        unvisited.remove(current)
    while unvisited:
        nxt = min(unvisited, key=lambda x: distmat[current][x])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    # return to start
    route.append(start_idx)
    return route

def two_opt(route, distmat, improvement_threshold=0.0001):
    best = route[:]
    improved = True
    best_distance = route_length(best, distmat)
    while improved:
        improved = False
        for i in range(1, len(best)-3):
            for j in range(i+1, len(best)-2):
                if j-i == 1: continue
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_dist = route_length(new_route, distmat)
                if new_dist + improvement_threshold < best_distance:
                    best = new_route
                    best_distance = new_dist
                    improved = True
        # loop until no improvement
    return best

def save_routes_to_csv(routes, points_df, depot_lat, depot_lon):
    # routes: dict driver_id -> list of point indices in order (including depot index 0)
    rows = []
    for drv, route in routes.items():
        for seq, idx in enumerate(route):
            if idx == 0:
                rows.append({
                    "driver": drv,
                    "seq": seq,
                    "id": "depot",
                    "lat": depot_lat,
                    "lon": depot_lon,
                    "address": "Depot"
                })
            else:
                src_idx = idx - 1  # adjust from all_points index to points_df index
                id_val = points_df.iloc[src_idx]["id"] if "id" in points_df.columns else src_idx
                lat_val = points_df.iloc[src_idx]["lat"]
                lon_val = points_df.iloc[src_idx]["lon"]
                addr_val = points_df.iloc[src_idx]["address"] if "address" in points_df.columns else ""
                rows.append({
                    "driver": drv,
                    "seq": seq,
                    "id": id_val,
                    "lat": lat_val,
                    "lon": lon_val,
                    "address": addr_val
                })
    return pd.DataFrame(rows)
def double_bridge_move(route):
    # route must start/end at depot (index 0). keep endpoints fixed.
    if len(route) < 8: 
        return route[:]  # too small
    n = len(route) - 1
    a, b, c, d = sorted(random.sample(range(1, n), 4))
    # 4-edge cut and reconnect (Lin–Kernighan style)
    part1 = route[1:a]
    part2 = route[a:b]
    part3 = route[b:c]
    part4 = route[c:d]
    tail  = route[d:n]
    new_mid = part1 + part3 + part2 + part4 + tail
    return [0] + new_mid + [0]

def simulated_annealing(route, distmat, T0=1.0, alpha=0.995, iters=1500):
    # safeguard: too few nodes → nothing to optimize
    if len(route) <= 3:
        return route[:]

    best = route[:]
    best_len = route_length(best, distmat)
    cur = route[:]
    cur_len = best_len
    T = T0

    for _ in range(iters):
        if len(cur) <= 3:
            break  # extra safety

        i, j = sorted(random.sample(range(1, len(cur)-1), 2))
        # 2-opt style neighbor (segment reversal)
        cand = cur[:i] + cur[i:j+1][::-1] + cur[j+1:]
        cand_len = route_length(cand, distmat)

        if cand_len < cur_len or random.random() < math.exp((cur_len - cand_len)/max(1e-9, T)):
            cur, cur_len = cand, cand_len
            if cur_len < best_len:
                best, best_len = cur, cur_len

        T *= alpha
        if T < 1e-6:
            T = 1e-6

    return best


def optimize_route_classical(cluster_indices, distmat):
    route = nearest_neighbor_tour(0, cluster_indices, distmat)
    route = two_opt(route, distmat)
    return route

def optimize_route_quantum(cluster_indices, distmat, effort=12, time_limit_s=1.5):
    # Multi-start + tunneling + SA + 2-opt
    deadline = time.time() + time_limit_s
    best = None
    best_len = float("inf")
    n = len(cluster_indices)
    # at least 'effort' restarts; also respect time limit
    restarts = 0
    while restarts < effort or time.time() < deadline:
        # random start
        route = [0] + random.sample(cluster_indices, n) + [0]
        # a few "quantum kicks" (double-bridge moves) to jump basins
        for _ in range(max(1, n//4)):
            route = double_bridge_move(route)
        # anneal in this basin
        route = simulated_annealing(route, distmat, T0=max(0.5, n/20), alpha=0.998, iters=1000 + 30*n)
        # local polish
        route = two_opt(route, distmat)
        l = route_length(route, distmat)
        if l < best_len:
            best, best_len = route, l
        restarts += 1
        if time.time() > deadline:
            break
    return best

# -------------------------
# UI - Sidebar
# -------------------------
st.sidebar.title("Mission Control")
st.sidebar.markdown("Upload CSV of deliveries. CSV columns: id (optional), lat, lon, address (optional).")

uploaded_file = st.sidebar.file_uploader("Drop CSV file here or click to upload", type=["csv"])

# Config
num_drivers = st.sidebar.number_input("Number of drivers / delivery boys", min_value=1, max_value=50, value=3)
optimization_mode = st.sidebar.radio("Optimization Mode", ("Classical", "Quantum (inspired)"))
quantum_effort = st.sidebar.slider("Quantum effort (restarts)", 4, 60, 16, help="Higher = slower but better")
quantum_time   = st.sidebar.slider("Quantum time cap (seconds)", 1.0, 8.0, 2.5)
depot_lat = st.sidebar.number_input("Depot latitude", value=15.8281, format="%.6f")
depot_lon = st.sidebar.number_input("Depot longitude", value=78.0373, format="%.6f")
fuel_l_per_km = st.sidebar.number_input("Fuel consumption (L/km)", min_value=0.01, value=0.12, format="%.3f")
co2_per_liter = st.sidebar.number_input("CO₂ per liter fuel (kg/L)", min_value=0.1, value=2.31, format="%.3f")
add_random_traffic = st.sidebar.checkbox("Add random traffic jams (demo)", value=False)
n_random_jams = st.sidebar.slider("Random traffic jams", 0, 10, 2)

# Buttons
run_button = st.sidebar.button("Launch Optimization")

# -------------------------
# Main UI - layout
# -------------------------
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.header("Delivery Console")
    st.markdown("Preview CSV and cluster assignments.")
    sample_csv = """id,lat,lon,address
1,15.8281,78.0373,Depot
2,15.8300,78.0350,Customer A
3,15.8250,78.0400,Customer B
4,15.8400,78.0300,Customer C
5,15.8200,78.0500,Customer D
"""
    st.markdown("Sample CSV format (click to copy) :")
    st.code(sample_csv, language="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if "lat" not in df.columns or "lon" not in df.columns:
                st.error("CSV must contain 'lat' and 'lon' columns.")
                st.stop()
            df = df.reset_index(drop=True)
            if "id" not in df.columns:
                df["id"] = df.index.astype(str)
            if "address" not in df.columns:
                df["address"] = ""
            st.dataframe(df.head(200))
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        st.info("No CSV uploaded. You can test using sample CSV content above by saving and uploading it.")
        df = None

    st.markdown("---")
    st.subheader("Optimization Timeline")
    st.markdown("Step 1: Orders clustered.  Step 2: Routes optimized.  Step 3: Real-time re-routing (traffic)")

with col2:
    st.header("Map — Optimization View")
    # Setup map
    map_center = [depot_lat, depot_lon]
    m = folium.Map(location=map_center, zoom_start=13, tiles="CartoDB dark_matter")

    # Add depot marker
    folium.CircleMarker(location=map_center, radius=8, color="cyan", fill=True, fill_color="cyan",
                        tooltip="Depot").add_to(m)

    # If no CSV, show only map
    if df is None:
        st_folium(m, width=900, height=700)
    else:
        # Prepare points list
        points = [(float(r.lat), float(r.lon)) for r in df.itertuples()]
        # Optionally add random traffic jams
        traffic_jams = []
        if add_random_traffic:
            rng = np.random.RandomState(42)
            for i in range(n_random_jams):
                # sample within bounding box of points
                lats = [p[0] for p in points] + [depot_lat]
                lons = [p[1] for p in points] + [depot_lon]
                lat = float(rng.uniform(min(lats), max(lats)))
                lon = float(rng.uniform(min(lons), max(lons)))
                traffic_jams.append((lat, lon))
                folium.CircleMarker(location=(lat, lon), radius=6, color="orange", fill=True, fill_color="orange",
                                    tooltip="Traffic Jam").add_to(m)

        # Cluster points using KMeans (cluster count = num_drivers)
        coords = np.array(points)
        k = min(num_drivers, len(points))
        if k <= 0:
            st.error("No points to optimize.")
            st.stop()

        kmeans = KMeans(n_clusters=k, random_state=0).fit(coords)
        labels = kmeans.labels_
        df["cluster"] = labels

        # show cluster markers
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        for i, r in df.iterrows():
            folium.CircleMarker(location=(r.lat, r.lon),
                                radius=6,
                                color=palette[int(r.cluster) % len(palette)],
                                fill=True,
                                fill_color=palette[int(r.cluster) % len(palette)],
                                tooltip=f"{r.id} (cluster {int(r.cluster)})").add_to(m)

        # Build distance matrix including depot as index 0
        all_points = [(depot_lat, depot_lon)] + points
        distmat = pairwise_distance_matrix(all_points, traffic_jams=traffic_jams)

        # For each cluster, compute route. We'll map cluster -> indices in all_points (with +1 offset)
        routes_by_driver = {}
        total_optimized_distance = 0.0
        baseline_total_distance = 0.0

        # baseline naive: sequential assignment of points to drivers by cluster order, route is depot->everypt->depot without optimization
        # We'll compute baseline per cluster as depot -> cluster points in given order -> depot
        for cluster_id in range(k):
            cluster_indices_local = [i for i, lbl in enumerate(labels) if lbl == cluster_id]  # indices in points (0-based)
            cluster_indices = [i+1 for i in cluster_indices_local]  # convert to all_points index
            if not cluster_indices:
                routes_by_driver[f"driver_{cluster_id}"] = [0, 0]
                continue

            # Baseline route
            baseline_route = [0] + cluster_indices + [0]
            baseline_len = route_length(baseline_route, distmat)
            baseline_total_distance += baseline_len

            # Choose optimizer
            if optimization_mode == "Quantum (inspired)":
                final_route = optimize_route_quantum(cluster_indices, distmat, effort=quantum_effort, time_limit_s=quantum_time)
            else:
                final_route = optimize_route_classical(cluster_indices, distmat)

            routes_by_driver[f"driver_{cluster_id}"] = final_route
            total_optimized_distance += route_length(final_route, distmat)

        # Draw routes on map with polylines
        for i, (drv, route) in enumerate(routes_by_driver.items()):
            path_coords = [(all_points[idx][0], all_points[idx][1]) for idx in route]
            folium.PolyLine(locations=path_coords, weight=4, color=palette[i % len(palette)],
                            tooltip=f"{drv} ({len(route)-2} stops)").add_to(m)
            # label driver at centroid of route
            mid_idx = len(path_coords)//2
            folium.map.Marker(path_coords[mid_idx],
                              icon=folium.DivIcon(html=f"""<div style="font-size:10px;color:{palette[i%len(palette)]}">● {drv}</div>""")
                              ).add_to(m)

        # Show optimized map
        st_folium(m, width=900, height=700)

        # Show metrics
        st.subheader("Optimization Results")
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Total distance (baseline)", f"{baseline_total_distance:.2f} km")
        with colB:
            st.metric("Total distance (optimized)", f"{total_optimized_distance:.2f} km")
        with colC:
            saved = max(0.0, baseline_total_distance - total_optimized_distance)
            st.metric("Distance reduced", f"{saved:.2f} km")

        # Estimate fuel and CO2 saved
        fuel_saved_l = saved * fuel_l_per_km
        co2_saved = fuel_saved_l * co2_per_liter
        st.write(f"Estimated fuel saved: **{fuel_saved_l:.2f} L**  (assumed {fuel_l_per_km:.3f} L/km)")
        st.write(f"Estimated CO₂ reduced: **{co2_saved:.2f} kg**  (assumed {co2_per_liter:.2f} kg CO₂ per L)")

        # Show per-driver stats
        st.markdown("**Per-driver route lengths (km)**")
        rows = []
        for drv, route in routes_by_driver.items():
            l = route_length(route, distmat)
            rows.append({"driver": drv, "stops": max(0, len(route)-2), "distance_km": round(l, 3)})
        st.table(pd.DataFrame(rows))

        # Download optimized routes CSV
        out_df = save_routes_to_csv(routes_by_driver, df.reset_index(drop=True), depot_lat, depot_lon)
        csv_bytes = out_df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv_bytes).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="optimized_routes.csv">Download optimized routes CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.info("Note: 'Quantum' mode here uses a quantum-inspired multi-restart + local search heuristic (simulated). "
                "If you later integrate real quantum solvers (D-Wave/IBM Qiskit), replace the route optimizer step with a QUBO solver.")

