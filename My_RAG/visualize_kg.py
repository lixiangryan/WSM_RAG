
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from pyvis.network import Network
    import networkx as nx
except ImportError:
    print("Error: Required libraries not installed. Please run 'pip install pyvis networkx'")
    sys.exit(1)

def visualize_kg(index_path="kg_index_zh.json", output_html="kg_visualization.html", max_nodes=200, min_weight=2):
    """
    Visualizes the Co-occurrence Graph with Community Detection and Dynamic Sizing.
    """
    if not os.path.exists(index_path):
        print(f"Error: Index file {index_path} not found.")
        return

    print(f"Loading KG index from {index_path}...")
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    co_occurrence_map = data.get("co_occurrence_map", {})
    if not co_occurrence_map:
        print("Error: No 'co_occurrence_map' found.")
        return

    # 1. Build NetworkX Graph (Weighted)
    G = nx.Graph()
    entity_strength = defaultdict(int)

    # Collect all edges first
    all_edges = []
    for src, neighbors in co_occurrence_map.items():
        for dst, weight in neighbors.items():
             if weight >= min_weight and src < dst:
                 all_edges.append((src, dst, weight))
                 entity_strength[src] += weight
                 entity_strength[dst] += weight

    # Sort nodes by total strength (Weighted Degree)
    sorted_nodes = sorted(entity_strength.items(), key=lambda x: x[1], reverse=True)
    top_nodes_list = [node for node, strength in sorted_nodes[:max_nodes]]
    top_nodes_set = set(top_nodes_list)
    
    # Add nodes to G
    for node in top_nodes_list:
        G.add_node(node, strength=entity_strength[node])
        
    # Add edges to G
    for src, dst, weight in all_edges:
        if src in top_nodes_set and dst in top_nodes_set:
            G.add_edge(src, dst, weight=weight)

    print(f"Graph constructed: {len(G.nodes)} nodes, {len(G.edges)} edges.")

    # 2. Community Detection (Louvain or Greedy Modularity)
    print("Detecting communities...")
    try:
        # returns list of sets of nodes
        communities = nx.community.greedy_modularity_communities(G, weight='weight')
        print(f"Detected {len(communities)} communities.")
    except Exception as e:
        print(f"Community detection failed: {e}. Fallback to single group.")
        communities = [set(G.nodes)]

    # Map node to community ID (for coloring)
    node_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i

    # Define a vibrant color palette (Category10-like)
    # https://d3js.org/d3-scale-chromatic/categorical
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
    ]

    # 3. Initialize PyVis Network
    net = Network(height="850px", width="100%", bgcolor="#1a1a1a", font_color="white", select_menu=True, filter_menu=True)
    
    # Enable Physics with ForceAtlas2Based (Good for clustering)
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
        overlap=0
    )

    # 4. Add Nodes to PyVis
    for node in top_nodes_list:
        strength = entity_strength[node]
        comm_id = node_community.get(node, 0)
        color = colors[comm_id % len(colors)]
        
        # Friendly Label
        label = node.split(":", 1)[1] if ":" in node else node
        
        # Dynamic Size (Log Scale or Sqrt Scale for better visual)
        # Base size = 10, add factor * sqrt(strength)
        size = 10 + (strength ** 0.5) * 2
        
        title_html = (
            f"<b>{label}</b><br>"
            f"Full ID: {node}<br>"
            f"Strength: {strength}<br>"
            f"Group: {comm_id}"
        )

        net.add_node(node, label=label, title=title_html, 
                     color=color, value=size, group=comm_id)

    # 5. Add Edges to PyVis
    for src, dst, data in G.edges(data=True):
        weight = data['weight']
        # Dynamic Width
        width = (weight ** 0.5) * 0.5
        # Opacity could be handled via color in rgba, but simple is good.
        
        net.add_edge(src, dst, value=width, title=f"Co-occurrence: {weight}", color="#666666")

    # 6. Save
    print(f"Saving visualization to {output_html}...")
    net.save_graph(output_html)
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="kg_index_zh.json", help="Path to KG index json")
    parser.add_argument("--output", default="kg_visualization.html", help="Output HTML file")
    parser.add_argument("--nodes", type=int, default=200, help="Max nodes to visualize")
    args = parser.parse_args()
    
    visualize_kg(args.index, args.output, max_nodes=args.nodes)
