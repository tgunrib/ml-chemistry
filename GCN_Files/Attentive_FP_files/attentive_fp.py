import pandas
from math import sqrt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP

# Train the model using the data without descriptors or clusters
from GCN_Files.Attentive_FP_files.pytorch_dataset import train_dataset, val_dataset, test_dataset

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=200)
test_loader = DataLoader(test_dataset, batch_size=200)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentiveFP(in_channels=39, hidden_channels=200, out_channels=1,
                    edge_dim=10, num_layers=2, num_timesteps=2,
                    dropout=0.2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -2.5,
                             weight_decay=10 ** -5)


def train():
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    print(sqrt(total_loss / total_examples))
    return sqrt(total_loss / total_examples)


@torch.no_grad()
def test(loader):
    mse = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
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
