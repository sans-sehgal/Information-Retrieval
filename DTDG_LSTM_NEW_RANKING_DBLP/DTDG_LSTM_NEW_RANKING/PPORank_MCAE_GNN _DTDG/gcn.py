from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import torch
from node2vec import Node2Vec
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from ge import DeepWalk
import networkx as nx
import pandas as pd
import pickle
import numpy as np

class GCN(torch.nn.Module):
	def __init__(self):
		super().__init__()
		""" GCNConv layers """
		self.linear1 = nn.Linear(128, 64)
		self.linear2 = nn.Linear(64, 1)
	def forward(self, nodes, edges):

		G = nx.DiGraph()
		G.add_nodes_from(nodes)
		G.add_edges_from(edges)
		# model = DeepWalk(G, walk_length=80, num_walks=10, workers=1)
		# model.train(window_size=5, iter=3)
		n2v = Node2Vec(G, dimensions=128, walk_length=20, num_walks=10, workers=1)
		n2v_model = n2v.fit(window=2, min_count=1, epochs=10, seed=123)
		embeddings = n2v_model.wv.vectors
		# print(embeddings.shape)
		x = self.linear1(torch.from_numpy(embeddings))
		#.to(device))
		x = F.relu(x)
		# return x
		torch.save(x , 'embeddings.pt')
		x = F.dropout(x, training=self.training)
		x = self.linear2(x)
		return torch.sigmoid(x)

def compute_accuracy(pred_y, y):
	return (pred_y == y).sum()

def embeddings(nodes , edges, y):

	# print("X shape: ", data.x.shape)
	# print("Edge shape: ", data.edge_index.shape)
	# print("Y shape: ", data.y.shape)

	model = GCN() #.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

	checkpoint = torch.load("model.pt")
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	# with open('labels' , 'rb') as f:
	# 	y = pickle.load(f)
	# train the model
	# y = y[:111]
	# criterion = nn.BCELoss()
	# model.train()
	# losses = []
	# accuracies = []
	# for epoch in range(2):
	# 	optimizer.zero_grad()
	# 	out = model(nodes,edges)
	# 	# predict = []
	# 	# for i,j in enumerate(y):
	# 	# 	predict.append(out[i][j])

	# 	# loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
	# 	# print(torch.tensor(y))
	# 	# print(out)
	# 	# out = torch.tensor(out.detach.numpy(out.reshape(1,-1)))
	# 	# print(out)


	# 	loss = criterion(out.squeeze(), torch.tensor(y).to(torch.float32))


	# 	# correct = compute_accuracy(out.argmax(dim=1)[data.train_mask], data.y[data.train_mask])
	# 	# acc = int(correct) / int(data.train_mask.sum())
	# 	losses.append(loss.item())
	# 	# out = torch.tensor(np.array(out.reshape(1,-1)))


	# 	# accuracies.append(acc)
	# 	loss.backward()
	# 	optimizer.step()
	# 	if (epoch+1) % 10 == 0:
	# 		print('Epoch: {}, Loss: {:.4f}, Training Acc: {:.4f}'.format(epoch+1, loss.item(), acc))


	torch.save({'epoch': 200,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, "model.pt")


	model.eval()
	pred=model(nodes, edges)
	# print(pred)
	# return pred

if __name__ == '__main__':
	embeddings()
