import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

data = torch.load('./data_simple.pt')
data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)
print(len(data.y[data.y == 0]), "0")
print(len(data.y[data.y == 1]), "1")


def oversample_minority(data, ratio=3):
    minority_indices = torch.where(data.y == 1)[0]
    minority_count = len(minority_indices)
    
    if minority_count == 0:
        print("No minority class samples found!")
        return data
    
    oversample_count = minority_count * ratio
    
    duplicate_indices = minority_indices[torch.randint(0, minority_count, (oversample_count,))]
    
    new_x = torch.cat([data.x, data.x[duplicate_indices]], dim=0)
    
    new_y = torch.cat([data.y, data.y[duplicate_indices]], dim=0)
    
    num_original_nodes = data.x.size(0)
    
    edge_index = data.edge_index
    new_edges = []
    
    for i, orig_idx in enumerate(duplicate_indices):
        new_node_idx = num_original_nodes + i
        
        mask_source = (edge_index[0] == orig_idx)
        mask_target = (edge_index[1] == orig_idx)
        
        targets = edge_index[1][mask_source]
        if len(targets) > 0:
            new_source_edges = torch.stack([
                torch.full_like(targets, new_node_idx),
                targets
            ])
            new_edges.append(new_source_edges)
        
        sources = edge_index[0][mask_target]
        if len(sources) > 0:
            new_target_edges = torch.stack([
                sources,
                torch.full_like(sources, new_node_idx)
            ])
            new_edges.append(new_target_edges)
    
    if new_edges:
        all_new_edges = torch.cat(new_edges, dim=1)
        new_edge_index = torch.cat([edge_index, all_new_edges], dim=1)
    else:
        new_edge_index = edge_index
    
    new_data = Data(x=new_x, edge_index=new_edge_index, y=new_y)
    
    return new_data

data = oversample_minority(data, ratio=10)


torch.save(data, './data_simple_oversampled.pt')
