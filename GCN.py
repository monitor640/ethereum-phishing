import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
data = torch.load('./data_simple_oversampled.pt', weights_only=False)


num_nodes = data.x.size(0)
node_indices = np.arange(num_nodes)

# Split node indices, not the features themselves
train_indices, temp_indices = train_test_split(node_indices, test_size=0.4, stratify=data.y.cpu().numpy())
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, stratify=data.y[temp_indices].cpu().numpy())

# Convert to tensors and move to appropriate device
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_indices] = True
test_mask[test_indices] = True
val_mask[val_indices] = True

y_train = data.y[train_mask].cpu().numpy()
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_classes=2, dropout=0.25):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv4(x, edge_index)
        

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(data.num_node_features).to(device)
data = data.to(device)
class_weights = class_weights.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-3)


model.train()
for epoch in range(400):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[train_mask], data.y[train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(data).argmax(dim=1)
            val_acc = (val_pred[val_mask] == data.y[val_mask]).float().mean()
            
            val_y_true = data.y[val_mask].cpu().numpy()
            val_y_pred = val_pred[val_mask].cpu().numpy()
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_y_true, val_y_pred, average='weighted', zero_division=0
            )
            f1_minority = f1_score(val_y_true, val_y_pred, pos_label=1, zero_division=0)

            # Count predictions
            pred_0 = (val_y_pred == 0).sum()
            pred_1 = (val_y_pred == 1).sum()
            
            print(f'Epoch {epoch:3d}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}, F1 Minority: {f1_minority:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            print(f'  Val predictions - 0: {pred_0}, 1: {pred_1}')
            
        model.train()


with torch.no_grad():
    model.eval()
    out = model(data)
    test_pred = out.argmax(dim=1)
    test_pred_proba = F.softmax(out, dim=1)
    
    # Get test set predictions and true labels
    y_true = data.y[test_mask].cpu().numpy()
    y_pred = test_pred[test_mask].cpu().numpy()
    y_pred_proba = test_pred_proba[test_mask].cpu().numpy()
    
    # Calculate accuracy and F1
    test_acc = (y_pred == y_true).mean()
    test_f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nGCN - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(f"  Predictions - Normal: {(y_pred == 0).sum()}, Phishing: {(y_pred == 1).sum()}")
    
    # Detailed classification report
    print("\n=== Detailed Classification Report ===")
    target_names = ['Normal', 'Phishing']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=2)
    print(report)
    
    # Confusion Matrix
    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    print(f"[[{cm[0,0]:5d} {cm[0,1]:5d}]")
    print(f" [{cm[1,0]:5d} {cm[1,1]:5d}]]")
    print("[[TN, FP],")
    print(" [FN, TP]]")
    
    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    sns.lineplot(x=fpr, y=tpr)
    plt.show()
    
    print(f"\nROC AUC Score: {roc_auc:.4f}")
