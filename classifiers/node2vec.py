from torch_geometric.nn import Node2Vec
import torch
import torch.multiprocessing
# Move graph to GPU if available
def run_node2vec():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load('../data_simple_oversampled.pt')
    # Define Node2Vec model
    node2vec = Node2Vec(
        edge_index=data.edge_index,  # Edge list
        embedding_dim=128,          # Embedding dimensionality
        walk_length=20,             # Length of random walks
        context_size=10,            # Context size for training
        walks_per_node=10,          # Number of walks per node
        num_negative_samples=1,     # Number of negative samples
        sparse=True                 # Use sparse embeddings for memory efficiency
    ).to(device)

    # Train the model
    loader = node2vec.loader(batch_size=128, shuffle=True, num_workers=0)  # Data loader
    optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=0.01)           # Optimizer

    def train():
        node2vec.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))  # Compute loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    # Train for multiple epochs
    for epoch in range(1, 20):
        loss = train()
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")

    # Get embeddings for all nodes
    node2vec_embeddings = node2vec(torch.arange(data.num_nodes, device=device))
    print("Node embeddings shape:", node2vec_embeddings.shape)

    return node2vec_embeddings

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    embeddings = run_node2vec()
    torch.save(embeddings, './node2vec_embeddings.pt')