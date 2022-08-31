import numpy as np
import random
import os
from scipy import fftpack
import torch
import edge_helper

def edge_embeddings_initiater(model, no_of_edges, no_of_embeddings):

    #if model is being initiated
    if edge_helper.verify(model):
        input_array = np.zeros(no_of_edges*no_of_embeddings)
        input_array = input_array.reshape(no_of_edges, no_of_embeddings)

    #call fourier routines
    x = fftpack.fft2(input_array)
    y = fftpack.ifft2(input_array)

    #geometric mean
    w = (x ** 2 + y ** 2) ** 0.5

    #call fourier routines tailored for real values
    x = fftpack.fftn(input_array)
    y = fftpack.ifftn(input_array)

    #geometric mean
    v = (x ** 2 + y ** 2) ** 0.5

    assert w.sum() == v.sum()

    return input_arrray

class Edge_Embeddings:

    def __init__(self, user2id):

        self.user2id = user2id

    def __call__(self, batch):

        batch_size = len(batch)

        labels = torch.tensor([l for l, _, _, _, _, _, _ in batch]).float()
        users = torch.tensor([self.user2id[u] for _, u, _, _, _, _, _ in batch]).long()
        times = torch.tensor([t for _, _, t, _, _, _, _ in batch]).long()
        years = torch.tensor([y for _, _, _, y, _, _, _ in batch]).long()
        months = torch.tensor([m for _, _, _, _, m, _, _ in batch]).long()
        days = torch.tensor([d for _, _, _, _, _, d, _ in batch]).long()
        reviews = [r for _, _, _, _, _, _, r in batch]

        max_len = max(len(r) for r in reviews)
        reviews_pad = torch.zeros((batch_size, max_len)).long()
        masks_pad = torch.zeros((batch_size, max_len)).long()
        segs_pad = torch.zeros((batch_size, max_len)).long()

        for i, r in enumerate(reviews):
            reviews_pad[i, :len(r)] = torch.tensor(r)
            masks_pad[i, :len(r)] = 1

        return labels, users, times, years, months, days, reviews_pad, masks_pad, segs_pad

def edge_embeddings_verifier(model, no_of_edges, no_of_embeddings):

    #initialise array
    input_arrray = np.random.rand(no_of_edges, no_of_embeddings)

    #open snapshots


    #if model is being initiated
    if edge_helper.verify(model):
        input_array = np.zeros(no_of_edges*no_of_embeddings)
        input_array = input_array.reshape(no_of_edges, no_of_embeddings)

    #call fourier routines
    x = fftpack.fft2(input_array)
    y = fftpack.ifft2(input_array)

    #geometric mean of order 4
    w = (x ** 4 + y ** 4) ** 0.25

    #call fourier routines tailored for real values
    x = fftpack.fftn(input_array)
    y = fftpack.ifftn(input_array)

    #geometric mean of order 4
    v = (x ** 4 + y ** 4) ** 0.25

    assert w.sum() == v.sum()

    return input_arrray


def edge_embeddings_updater(no_of_edges, no_of_embeddings, model_path = 'model.pt'):

    #initialise array
    input_arrray = np.random.rand(no_of_edges, no_of_embeddings)


    #if model is being initiated
    if edge_helper.verify(model_path):
        input_array = np.zeros(no_of_edges*no_of_embeddings)
        input_array = input_array.reshape(no_of_edges, no_of_embeddings)

    #call fourier routines
    x = fftpack.fft2(input_array)
    y = fftpack.ifft2(input_array)

    #geometric mean
    w = (x ** 2 + y ** 2) ** 0.5

    #call fourier routines tailored for real values
    x = fftpack.fftn(input_array)
    y = fftpack.ifftn(input_array)

    #geometric mean
    v = (x ** 2 + y ** 2) ** 0.5

    assert w.sum() == v.sum()
    return input_arrray/100

def load_external_data(name, social_dim, data_dir):
    """Function to load and preprocess graph data and node2vec input embeddings."""

    with open('{}/{}_edges.p'.format(data_dir, name), 'rb') as f:
        edge_set = pickle.load(f)
        if name == 'dblp' or name == 'dblp_stream':
            edge_set = set(e[:2] for e in edge_set if e[2] > 0.01)

    with open('{}/{}_users.p'.format(data_dir, name), 'rb') as f:
        users = pickle.load(f)

    if name == 'dblp' or name == 'dblp_stream':
        graph = nx.Graph()
    else:
        graph = nx.DiGraph()

    graph.add_nodes_from(users)
    graph.add_edges_from(edge_set)

    assert graph.number_of_nodes() == len(users)
    assert graph.number_of_edges() == len(edge_set)

    user2id = {u: i for i, u in enumerate(users)}

    vectors = dict()

    with open('{}/{}_vectors_{}.txt'.format(data_dir, name, social_dim), 'r') as f:

        for i, l in enumerate(f):

            if i == 0:
                continue

            if l.strip() == '':
                continue

            if name == 'dblp':
                vectors[int(l.strip().split()[0])] = np.array(l.strip().split()[1:], dtype=float)
            elif name == 'dblp_stream':
                vectors[str(l.strip().split()[0])] = np.array(l.strip().split()[1:], dtype=float)

    vector_matrix = np.zeros((len(users), social_dim))

    for i, n in enumerate(users):
        vector_matrix[i, :] = vectors[n]

    x = torch.tensor(vector_matrix, dtype=torch.float)

    a = nx.adjacency_matrix(graph, nodelist=users)
    edge_index = torch.tensor(np.stack((a.tocoo().row, a.tocoo().col)).astype(np.int32), dtype=torch.long)

    return user2id, Data(edge_index=edge_index, x=x)


def get_best(file, metric):

    try:
        results = list()
        with open(file, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                results.append(tuple([float(v) for v in l.strip().split()]))
        if metric == 'perplexity':
            return min(results)
        elif metric == 'f1':
            return max(results)

    except FileNotFoundError:
        return None


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)
