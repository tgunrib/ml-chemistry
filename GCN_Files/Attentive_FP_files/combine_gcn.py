# Define a non-graphical neural network for processing molecular descriptors
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader

from GCN_Files.Attentive_FP_files.pytorch_dataset import train_dataset, val_dataset, test_dataset

torch.autograd.set_detect_anomaly(True)

#Implements the model used to train the gnn with global features.

class GCNLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNLinear, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x


class CombinedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_classes, graph_input_dim,
                 edge_dim, num_layers):
        super(CombinedNN, self).__init__()
        self.non_graphical_nn = GCNLinear(input_dim, hidden_dim, num_classes)
        self.graphical_nn = AttentiveFP(in_channels=graph_input_dim,out_channels=num_classes,
                                        hidden_channels=hidden_dim,edge_dim=edge_dim,num_layers=num_layers,
                                        num_timesteps=2,dropout=0.2).to(device)
        self.fc = nn.Linear(2 * num_classes, num_classes)

    def forward(self, x, descriptors, edge_index, edge_attr, batch):
        # Pass the molecular descriptors through the non-graphical neural network
        non_graphical_output = self.non_graphical_nn.forward(descriptors)
        # Pass the molecular graphs through the AttentiveFP-based graphical neural network
        graphical_output = self.graphical_nn.forward(x, edge_index, edge_attr, batch)
        # Concatenate the non-graphical and graphical outputs
        combined_output = torch.cat((non_graphical_output, graphical_output), dim=1)

        # Pass the combined output through a fully connected layer
        final_output = self.fc(combined_output)
        return final_output


train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=200)
test_loader = DataLoader(test_dataset, batch_size=200)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = CombinedNN(input_dim=237,hidden_dim=200,graph_input_dim=39,num_layers=4,num_classes=1,edge_dim=10)
model = GCNLinear(input_dim=237,hidden_dim=512,output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -5,
                             weight_decay=10 ** -5)


def train():
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        #out = model.forward(data.x, data.descriptors, data.edge_index, data.edge_attr, data.batch)
        out = model.forward(data.descriptors)
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
        out = model.forward(data.descriptors)
        #out = model.forward(data.x, data.descriptors, data.edge_index, data.edge_attr, data.batch)
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
output.to_csv("AttentiveFP_only_descriptors.csv")



