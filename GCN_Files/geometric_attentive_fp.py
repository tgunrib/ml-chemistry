from math import sqrt

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch_geometric import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP

from GCN_Files.featuriser_graph import create_pytorch_geometric_graph_data_list_from_smiles_and_labels
# Uses the featuriser_graph to train the model

if __name__ == '__main__':
    # canonical training loop for a Pytorch Geometric GNN model gnn_model
    data = pd.read_csv(input())
    x_smiles = list(data.SMILES)
    y = list(data.logS)
    # create list of molecular graph objects from list of SMILES x_smiles and list of labels y
    data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y)
    # create dataloader for training
    dataloader = DataLoader(dataset=data_list, batch_size=200)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentiveFP(in_channels=data_list[0].num_features, hidden_channels=200, out_channels=1,
                        edge_dim=10, num_layers=2, num_timesteps=2,
                        dropout=0.2).to(device)

    # define loss function
    loss_function = nn.MSELoss()
    # define optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    accuracy = {}
    # loop over 10 training epochs
    for epoch in range(250):
        # set model to training mode
        model.train()
        total_loss = total_examples = 0
        # loop over minibatches for training
        for (k, data) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = F.mse_loss(out.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
            total_examples += data.num_graphs
        accuracy[epoch] = sqrt(total_loss / total_examples)
        print(sqrt(total_loss / total_examples))

    epochs = list(accuracy.keys())
    rmse = list(accuracy.values())
    plt.plot(epochs,rmse)
    plt.savefig("epoch_rmse_geometric")
