# Information-Retrieval

### ABSTRACT 
<div align="justify"> Finding pertinent information from a variety of sources is known as information retrieval (IR). Since
the invention of computers, information retrieval systems that support and accomplish this goal have
been of highest importance. Their significance has multiplied with the advent of the Internet and web
search engines. The Learning to Rank (LTR) subtask, which deals with the ranking of information
retrieved based on multiple parameters including relevance ratings, user feedback, and citations is an
essential part of information retrieval systems. More recently, the "Learning to Rank" problem has
been resolved and improved using deep Reinforcement Learning (RL) algorithms (systems that harness
the strength of neural nets and the soundness of reinforcement learning paradigms). Research has
shown multiple deep Reinforcement Learning-based algorithms have been employed to perform the
Learning to Rank task and their efficacy is evident when evaluated on standard Learning to Rank
benchmark datasets.

In order to build upon previous research, achieve our objectives, and to improve LTR models, we
incorporate a framework that utilises Graph Neural Networks along with Reinforcement Learning. We
use citation networks to incorporate GNNs into our research. This is followed by the utilisation of RL
algorithms such as PPO and MDPRank to rank the documents present in the citation network. The
entire workflow is explained in much more detail later in the report.
</div>

### METHODOLOGY

<div align = "justify"> We use RL algorithms for ranking while utilising Graph Neutral Networks or GNN’s to generate node
embeddings. The first step in the process is calculating the node embeddings of documents. Libraries
such as Node2Vec and DeepWalk are used to calculate embeddings of the nodes. In our problem
statement, the nodes are the documents that need to be ranked and the edges are the links between the
documents. For instance, in the scenario of searching for research papers, the nodes will be the papers
themselves and the links will be the citations these papers carry to different papers within the dataset.
<br>
<br>
<br>

![Node Embeddings](https://github.com/sans-sehgal/Information-Retrieval/blob/main/Images/Node%20Embeddings.png)
<br>
<br>
<br>
![Skip Gram Model](https://github.com/sans-sehgal/Information-Retrieval/blob/main/Images/Skip%20Gram%20Model.png)
 <br>
 <br>
 <br>
Initially, we carry out these rankings on a static graph, and analyse the results obtained. However, since
static graphs involve training the GNN model from scratch every time new nodes (aka documents) are
added, it takes too long to train. Thus, to make our implementation more practical and feasible to real
world systems, we attempt to use dynamic graphs.

We try to perform these ranking dynamically, using Dynamic Graphs and simulating a real-world
environment. We start with a small number of nodes and edges and use a ranking algorithm such as
Proximal Policy Optimization to rank these nodes. The model in this case is not retrained from scratch,
but rather reloaded from its previous state to ensure quick convergence.
</div>

### STATIC GRAPHS
<div align = "justify"> In this setting, we use a static GNN to rank documents based on the embeddings generated using
Node2Vec. Since we use a static GNN, the model is trained from scratch each time new nodes are
added into the system.
<br>
<br>
</div>
<div align = "justify">  Dataset: We use the CORA dataset for preliminary testing. Once we were able to obtain good results
from this dataset, we continue our testing on larger datasets such as DBLP which will prove the
reliability of the system. Testing on the latter is currently underway. The CORA dataset consists of
2708 Nodes (or documents) while the DBLP dataset consists of 3 million nodes.
</div>
