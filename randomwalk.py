import pickle 
import networkx as nx
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

G = load_pickle('./MulDiGraph.pkl')


def extract_phishing_subgraph(G, walk_length=100, num_walks_per_node=5):
    fraud_nodes = [node for node in G.nodes if G.nodes[node]['isp'] == 1]
    
    subgraph_nodes = set(fraud_nodes)
    
    for fraud_node in fraud_nodes:
        for _ in range(num_walks_per_node):
            walk = perform_random_walk(G, fraud_node, walk_length)
            subgraph_nodes.update(walk)
    subgraph_view = G.subgraph(subgraph_nodes)
    subgraph_copy = subgraph_view.copy()
    return subgraph_copy


def perform_random_walk(G, start_node, walk_length):
    walk = [start_node]
    current_node = start_node
    
    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(current_node))
        
        if not neighbors:
            break
        current_node = random.choice(neighbors)
        walk.append(current_node)
    
    return walk

if __name__ == "__main__":
    random.seed(42)
    G = extract_phishing_subgraph(G, walk_length=300, num_walks_per_node=150)
    print(f"Final Subgraph - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    save_pickle(G, './phishing_subgraph.pkl')
