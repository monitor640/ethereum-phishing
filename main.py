import pickle 
import networkx as nx
import numpy as np

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

G = load_pickle('./MulDiGraph.pkl')
#print(nx.info(G))

# Traversal nodes:
for idx, nd in enumerate(nx.nodes(G)):
    print(G.nodes[nd])
    break

# Travelsal edges:
for ind, edge in enumerate(nx.edges(G)):
    (u, v) = edge
    eg = G[u][v][0]
    amo, tim = eg['amount'], eg['timestamp']
    print(u, v, amo, tim)
    break


# Count labeled nodes
labeled_nodes = sum(1 for n in G.nodes() if G.nodes[n].get('isp', 0) == 1)
print(f"Labeled phishing nodes: {labeled_nodes} out of {G.number_of_nodes()}")

# Transaction amount statistics
amounts = [G[u][v][0]['amount'] for u, v in G.edges()]
print(f"Average transaction amount: {np.mean(amounts)}")
print(f"Max transaction: {np.max(amounts)}")
print(f"Min transaction: {np.min(amounts)}")

# Temporal analysis
timestamps = [G[u][v][0]['timestamp'] for u, v in G.edges()]
time_range = max(timestamps) - min(timestamps)
print(f"Transaction time span: {time_range} seconds ({time_range/86400:.2f} days)")

#find the first and last transaction date
import datetime
first_transaction = datetime.datetime.fromtimestamp(min(timestamps)).strftime('%Y-%m-%d')
last_transaction = datetime.datetime.fromtimestamp(max(timestamps)).strftime('%Y-%m-%d')
print(f"First transaction date: {first_transaction}")
print(f"Last transaction date: {last_transaction}")

#see if there are ary nodes that have no trasactions with nodes where the  ['isp'] is 1
phishing_nodes = set(n for n in G.nodes() if G.nodes[n].get('isp', 0) == 1)

# Find nodes that haven't transacted with phishing nodes
nodes_no_phishing_contact = set()

for node in G.nodes():
    neighbors = set(G.predecessors(node)) | set(G.successors(node))
    if not neighbors & phishing_nodes:
        nodes_no_phishing_contact.add(node)

print(f"\nNodes with no transactions with phishing accounts: {len(nodes_no_phishing_contact)}")
print(f"Percentage: {(len(nodes_no_phishing_contact) / G.number_of_nodes()) * 100:.2f}%")



