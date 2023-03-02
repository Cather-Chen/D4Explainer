import networkx as nx
import numpy as np
from grandiso import find_motifs
import matplotlib.pyplot as plt
import scipy.special
import time
import json
from torch_geometric.utils import to_networkx
from utils.dataset import get_datasets
plt.rcParams["figure.figsize"] = (12,8)
plt.rcParams.update({'font.size': 12})

def nb_samples_required(a, delta, epsilon):
    return int(np.ceil(2 * (np.log(2) * a + np.log(1 / delta)) / (epsilon**2)))


def build_graphlets(k):
    if k == 3:
        graphlets = [
            np.zeros((3, 3)),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        ]
    if k == 4:
        graphlets = [
            np.zeros((4, 4)),
            np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
            np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]),
            np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]),
            np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]),
            np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]]),
            np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]]),
            np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]),
            np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
        ]

        # np.array([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]),

    if k == 5:
        graphlets = [
            np.zeros((5, 5)),
            np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),  # 2
            np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),  # 3
            np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]),  # 4
            np.array([[0, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),  # 5
            np.array([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),  # 6
            np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]),  # 7
            np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]]),  # 8
            np.array([[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]),  # 9
            np.array([[0, 1, 1, 1, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),  # 10
            np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]),  # 11
            np.array([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]]),  # 12
            np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0]]),  # 13
            np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]]),  # 14
            np.array([[0, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]),  # 15
            np.array([[0, 1, 1, 0, 1], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0]]),  # 16
            np.array([[0, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]),  # 17
            np.array([[0, 1, 1, 1, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0]]),  # 18
            np.array([[0, 1, 0, 1, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [1, 0, 0, 0, 0]]),  # 19
            np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]]),  # 20
            np.array([[0, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 0, 0], [1, 0, 0, 0, 0]]),  # 21
            np.array([[0, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0]]),  # 22
            np.array([[0, 1, 3, 1, 0], [1, 0, 1, 1, 0], [1, 1, 0, 1, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]),  # 23
            np.array([[0, 1, 0, 1, 1], [1, 0, 1, 1, 0], [0, 1, 0, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]]),  # 24
            np.array([[0, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0]]),  # 25
            np.array([[0, 1, 0, 1, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0]]),  # 26
            np.array([[0, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 1, 0, 1, 1], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0]]),  # 27
            np.array([[0, 1, 3, 1, 0], [1, 0, 1, 1, 1], [1, 1, 0, 1, 0], [1, 1, 1, 0, 0], [0, 1, 0, 0, 0]]),  # 28
            np.array([[0, 1, 1, 1, 1], [1, 0, 1, 1, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 0, 0, 1, 0]]),  # 29
            np.array([[0, 1, 1, 0, 1], [1, 0, 1, 1, 0], [1, 1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 0, 0, 1, 0]]),  # 30
            np.array([[0, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0]]),  # 31
            np.array([[0, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0]]),  # 32
            np.array([[0, 1, 1, 1, 1], [1, 0, 1, 1, 0], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 0, 1, 1, 0]]),  # 33
            np.array([[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]]),  # 33
        ]

    return [nx.from_numpy_matrix(m) for m in graphlets]


def sample_subgraphs(G, k):
    nodes = G.nodes()

    return G.subgraph(np.random.choice(nodes, k, replace=False).tolist())


def find_iso_graphlet(subgraph, graphlets, spectrum):
    for i, graphlet in enumerate(graphlets):
        if nx.is_isomorphic(subgraph, graphlet):
            spectrum[i] += 1


def estimate_spectrum(G, k, delta, epsilon):
    graphlets = build_graphlets(k)
    m = nb_samples_required(len(graphlets), delta, epsilon)
    spectrum = np.zeros(len(graphlets))

    for _ in range(m):
        subgraph = sample_subgraphs(G, k)
        find_iso_graphlet(subgraph, graphlets, spectrum)

    #return spectrum / np.linalg.norm(spectrum) # to sum to 1 for kernel(G, G)
    return spectrum / np.sum(spectrum)


def kernel(G1, G2, k, delta, epsilon):
    spectrum1 = estimate_spectrum(G1, k, delta, epsilon)
    spectrum2 = estimate_spectrum(G2, k, delta, epsilon)

    return spectrum1.T @ spectrum2

name = "BA_Community"
train, val, test = get_datasets(name=name, root="data")
graph0 = train[0]
G0 = to_networkx(graph0, to_undirected=True)
feat_0 = estimate_spectrum(G0, 5, 0.05, 0.05)
print(graph0.self_y)
self = []
self.append(feat_0)
positive = []
negtive = []
print(len(test))
for i in range(len(test)):
    print(i)
    graph = test[i]
    G = to_networkx(graph, to_undirected=True)
    feat = estimate_spectrum(G, 5, 0.05, 0.05)
    if graph.self_y == graph0.self_y:
        positive.append(feat)
    else:
        negtive.append(feat)
doc = {"self": self, "neg": negtive, "pos": positive}
np.save(f'{name}.npy', doc)




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
doc_ = np.load(f'{name}.npy',allow_pickle=True).item()

self = doc_['self']
positive = doc_["pos"]
negtive = doc_['neg']
SameClass = []
DiffClass = []
for vec in positive:
    SameClass.append(np.sum(np.abs(vec-self)))
for vec in negtive:
    DiffClass.append(np.sum(np.abs(vec-self)))
Class = ["same"] * len(SameClass) + ["different"] * len(DiffClass)
dist = SameClass + DiffClass
plot_dict = {"Class": Class, "Distance": dist}
data = pd.DataFrame(list(plot_dict.items()),columns=['Class', 'Distance'])
sns.kdeplot(SameClass,shade=True,color="g")
sns.kdeplot(DiffClass,shade=True,color="b")
# plt.legend(labels=['Same Class','Different Class'], facecolor='white')
# plt.title("BA-3Motif Graphlet Distance Distribution")
plt.show()