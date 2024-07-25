### Import libraries and mounted at the folder contains additional required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch_geometric.utils import dense_to_sparse , _to_dense_adj
from torch_geometric_temporal.signal import StaticGraphTemporalSignal , temporal_signal_split

data = pd.read_csv("Clean-Return.csv")
data = data.drop(columns = ['Unnamed: 0', 'Date'], axis= 1, inplace = False)


### Create adjacency matrix
class GraphStructure:
    def __init__(self,
                 fund_distances: np.ndarray,
                 adj_matrix_threshold: float,
                 adj_matrix_keep_original: bool = False):
        self.fund_distances = fund_distances
        self.adj_matrix_threshold = adj_matrix_threshold
        self.adj_matrix_keep_original = adj_matrix_keep_original
        self.edges, self.num_nodes = self.compute_edge_node()

    def compute_adj_matrix(self):
        if self.adj_matrix_keep_original:
            return self.fund_distances
        ###### top-k relative matrix (new)
        k = int(self.fund_distances.shape[0]*0.01)
        top_k_indices = np.argsort(self.fund_distances.values, axis=1)[:, -k:]
        mask = np.zeros(self.fund_distances.shape, dtype=int)
        np.put_along_axis(mask, top_k_indices, 1, axis=1)
        return mask

        # Old method
        # num_funds = self.fund_distances.shape[0]
        # w_mask = np.ones([num_funds, num_funds]) - np.identity(num_funds)
        # return ((self.fund_distances >= self.adj_matrix_threshold) * w_mask).astype(int)

    def compute_edge_node(self):
        adj_matrix = self.compute_adj_matrix()
        node_indices, neighbor_indices = np.where(adj_matrix == 1)

        edges=node_indices.tolist(), neighbor_indices.tolist()
        num_nodes=adj_matrix.shape[0]

        return (edges, num_nodes)
    

### Create DataLoader
class DatasetLoader(object):

    def __init__(self, values_matrix, adj_matrix):
        super(DatasetLoader, self).__init__()
        self.values_matrix = values_matrix
        self.adj_matrix = adj_matrix
        A = adj_matrix
        X = values_matrix.permute(1, 2, 0)

        means = torch.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = torch.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)

        self.A = A
        self.X = X

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.adj_matrix)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for METR-LA dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The METR-LA traffic
                forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset
    

### Create input for model:
corr_matrix = 1-data.corr()
graph = GraphStructure(corr_matrix, adj_matrix_threshold=0.25)
print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")
adj_matrix = graph.compute_adj_matrix()
adj_matrix = torch.tensor(adj_matrix)
X_matrix = torch.unsqueeze(torch.tensor(data.values), dim = 2)

### Create dataset from loaded csv, timesteps_out = 18
loader = DatasetLoader(X_matrix,adj_matrix)
dataset = loader.get_dataset(num_timesteps_in=18, num_timesteps_out=18)

### Split train-test dataset
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)


### Create model
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features,
                           out_channels=32,
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h


epochs = 20
# GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda

# Create model and load from save point
model = TemporalGNN(node_features=1, periods=18).to(device)
# model.load_state_dict(torch.load('/content/drive/MyDrive/GNN/model_weights.pth', map_location=device))

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
model.train()

print("Running training...")
for epoch in range(epochs):
    loss = 0
    step = 0
    for snapshot in train_dataset:
        snapshot = snapshot.to(device)
        # Get model predictions
        y_hat = model(snapshot.x, snapshot.edge_index)
        # Mean squared error
        loss = loss + torch.mean((y_hat-snapshot.y)**2)
        step += 1
    loss = loss / (step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))

torch.save(model.state_dict(), 'model_weights_fh18.pth')