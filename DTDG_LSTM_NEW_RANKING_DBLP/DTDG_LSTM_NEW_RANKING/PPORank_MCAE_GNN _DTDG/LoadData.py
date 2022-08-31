import numpy as np
from math import sqrt
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
# from Embeddings import *
from gcn import *
import random
from edge_embeddings import *

def normalize(features):
    """L2 Normalization on input and returned"""
    norm = np.linalg.norm(features)
    if norm == 0:
        return features
    else:
        return features / norm


def mean(income):
    mn = 0.0
    for i in income:
        mn += i[0]
    return round(mn / len(income), 4)


def std_hand(income):
    income -= round(mean(income), 4)
    sump = 0.0
    for i in income:
        print(i[0], end=' ')
        sump += round(i[0] * i[0], 4)
        print(sump)
    return round(sqrt(sump / len(income)), 4)


def normalize_std(income):
    """Mean Normalization on input and returned"""
    print(f"Mean = {np.mean(income)} Std = {np.round(np.std(income))}")
    print(f"Mean hand = {mean(income)} STD hand = {std_hand(income)}")
    income -= mean(income)
    print(income)
    std_h = std_hand(income)
    if std_h == 0:
        return np.round(income, 4)
    return np.round((income / std_h), 4)


def normalize_minmax(features):
    """Min Max Scaling done on input and returned"""
    if np.max(features) - np.min(features) == 0:
        return features
    return (features - np.min(features)) / (np.max(features) - np.min(features))


def import_dataset(filename, n_features):
    """Imports file into a structure:{qid:{docid:(features,label),docid2:(features_2,label_2)},qid2:{docid:(),docid2:()}} """
    f = open(filename, "r")
    data = {}
    genid = 0
    for line in f.readlines():
        x = line.split(" ")
        label = int(x[0])
        qid = int(x[1].split(":")[1])
        features = []
        for i in range(n_features):
            features.append(float(x[2 + i].split(":")[1]))
        #         features = normalize(features)
        if "#docid" in x:
            docid = x[x.index("#docid") + 2]
        else:
            docid = str(genid + 1)
            genid += 1
        if qid not in data:
            data[qid] = {docid: (features, label)}
        else:
            data[qid][docid] = (features, label)
    return data


def import_all(filename):
    f = open(filename, 'rb')
    t = pickle.load(f)
    s = pd.Series(t)
    training_data, test_data = [i.to_dict() for i in train_test_split(s, train_size=0.7, random_state=4)]
    return training_data, test_data

def get_dataset():
    with open('dblp_nodes_dtdg.p', 'rb') as f:
        nodes = pickle.load(f)
    with open('dblp_edges_dtdg.p', 'rb') as f:
        edges = pickle.load(f)
    return nodes , edges

def generate_dataset(prev_shape, nodes, edges, i):
    node_seq = set()
    edge_seq = set()
    query_seq = set()
    random.seed(7)
    for node in nodes:
        node_seq.add(node[1])
        for edge in edges:
            if (node[1]) == (edge[0]) or (node[1]) == (edge[1]):
                node_seq.add(edge[0])
                node_seq.add(edge[1])
                edge_seq.add(edge)
        if len(node_seq) > prev_shape + 100 or len(node_seq) >= 20819005:
            # edges_input_model =  pd.DataFrame(edge_seq)
            # edges_input_model = torch.from_numpy(edges_input_model.values)
            # edges_input_model = torch.transpose(edges_input_model,0,1)
            # edges_input_model = edges_input_model.type(dtype = torch.int64)
            if(i%5 == 0):
                with open('latest_snapshot/nodes_snapshot.p' , 'wb') as f:
                    pickle.dump(node_seq , f)
            y=[]
            for _ in range(len(node_seq)):
                y.append(random.randint(0, 1))
            # print(len(node_seq))
            # print(node_seq)
            # print(edge_seq)
            embeddings(node_seq, edge_seq, y)
            array_nodes = torch.load('embeddings.pt')
            array_nodes = array_nodes.detach().numpy()
            array_edges = edge_embeddings_updater(no_of_edges = len(node_seq),no_of_embeddings = 64)
            array = np.concatenate((array_nodes, array_edges) , axis = 1)

            for i in node_seq:
                for j in nodes:
                    if i == j[1]:
                        query_seq.add(j)
                        break
            final_dict = format_dataset(query_seq,node_seq,array)
            s = pd.Series(final_dict)
            training_data, test_data = [i.to_dict() for i in train_test_split(s, train_size=0.7, random_state=4)]
            prev_shape = len(node_seq)
            return training_data, test_data, prev_shape

'''
query_seq is -> query , docid, relevance score
node_seq is -> just doc id's
array is -> embeddings
'''
def format_dataset(query_seq,node_seq,array):
    final_dict = {}
    temp_dict = {}
    embeddings_dict = {}
    # embeddings_dict = array
    print(array.shape)
    j = 0
    for i,j in enumerate(node_seq):
        try:
            embeddings_dict[j] = array[i]
        except Exception as e:
            # for q in query_seq:
            #     if q[1] == j:
            #         query_seq.remove(q)
            pass

    # print(j)
    # print(embeddings_dict[j])
    query_seq = sorted(query_seq)

    for i in range(len(query_seq)):
        if i == len(query_seq)-1:
            try:
                temp_dict[query_seq[i][1]] = (list(embeddings_dict[query_seq[i][1]]) , query_seq[i][2])
                final_dict[query_seq[i][0]] = temp_dict
            except Exception as e:
                print(e)

        elif query_seq[i][0] == query_seq[i+1][0]:
            try:
                temp_dict[query_seq[i][1]] = (list(embeddings_dict[query_seq[i][1]]) , query_seq[i][2])
            except Exception as e:
                print(e)
        else:
            try:
                temp_dict[query_seq[i][1]] = (list(embeddings_dict[query_seq[i][1]]) , query_seq[i][2])
                final_dict[query_seq[i][0]] = temp_dict
                temp_dict = {}
            except Exception as e:
                print(e)
    return final_dict
