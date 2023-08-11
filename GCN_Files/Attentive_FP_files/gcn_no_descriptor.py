import pandas
from math import sqrt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric import nn
from torch_geometric.loader import DataLoader
from GCN_Files.Attentive_FP_files.pytorch_dataset import train_dataset, test_dataset, val_dataset
from GCN_Files.Attentive_FP_files.combine_gcn import CombinedNN

# Trains the model with global clusters and taxonomy
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=200)
test_loader = DataLoader(test_dataset, batch_size=200)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CombinedNN(input_dim=33,hidden_dim=200,graph_input_dim=39,num_layers=4,num_classes=1,edge_dim=10)

optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -5,
                             weight_decay=10 ** -5)


def train():
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model.forward(data.x, data.no_descriptors, data.edge_index, data.edge_attr, data.batch)
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(out,data.y)+1e-5)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
        print(loss)
    print(sqrt(total_loss / total_examples))
    return sqrt(total_loss / total_examples)


@torch.no_grad()
def test(loader):
    mse = []
    for data in loader:
        data = data.to(device)
        out = model.forward(data.x, data.no_descriptors, data.edge_index, data.edge_attr, data.batch)
        mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
    return float(torch.cat(mse, dim=0).mean().sqrt())


nums = np.arange(1, 201)
print(nums)
train_accuracy = []
val_accuracy = []
test_accuracy = []
for epoch in nums:
    train_rmse = train()
    val_rmse = test(val_loader)
    test_rmse = test(test_loader)
    train_accuracy.append(train_rmse)
    val_accuracy.append(val_rmse)
    test_accuracy.append(test_rmse)
    print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f} Val: {val_rmse:.4f} '
          f'Test: {test_rmse:.4f}')
output = pandas.DataFrame(zip(nums, train_accuracy, val_accuracy, test_accuracy),
                          columns=['Epoch', 'Train', 'Validation', 'Test'])
print(output)
output.to_csv("AttentiveFP.csv")