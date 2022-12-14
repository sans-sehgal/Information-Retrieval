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
<br>
<br>
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

![Node Embeddings](https://github.com/sans-sehgal/Information-Retrieval/blob/main/Images/Node%20Embeddings.png "Node2Vec")
<br>
<br>
<br>
![Skip Gram Model](https://github.com/sans-sehgal/Information-Retrieval/blob/main/Images/Skip%20Gram%20Model.png "Node2Vec Working")
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
Dataset: We use the <a href="https://relational.fit.cvut.cz/dataset/CORA"> Cora Dataset </a> for preliminary testing. Once we were able to obtain good results from this dataset, we continue our testing on larger datasets such as <a href = "https://www.aminer.org/citation">DBLP</a> which will prove the
reliability of the system. Testing on the latter is currently underway. The CORA dataset consists of
2708 Nodes (or documents) while the DBLP dataset consists of 3 million nodes.
</div>
<br>
<br>
<br>
<img src = "https://github.com/sans-sehgal/Information-Retrieval/blob/main/Images/static%20graph%20workflow.png"></img>
<br>


<h3> DYNAMIC GRAPHS </h3>
<div align = "justify">
The main advantage of dynamic graphs over static graphs is that we can save the model after every
iteration and reload it from its previous state. This technique allows the model to converge much faster
than when we initialise the model parameters from scratch as is the case with static graphs.
<br>
<br>
 Dataset: We use the <a href="https://relational.fit.cvut.cz/dataset/CORA"> Cora Dataset </a> dataset for preliminary testing. Once we were able to obtain good results
from this dataset, we continue our testing on larger datasets such as <a href = "https://www.aminer.org/citation">DBLP</a> which will testify to the
robustness of the system. The CORA dataset consists of 2708 Nodes (or documents) while the DBLP
dataset consists of 3 million nodes.
 <br>
 <br>
 <br>
<img src = "https://github.com/sans-sehgal/Information-Retrieval/blob/main/Images/dynamic%20graph%20workflow.png"></img>
<br>
<br>
<br>
<h3> RESULTS </h3>

This section describes the results obtained by each of the algorithms in the context of learning to rank.
Then analysis and comparison of these results with the state-of-the-art MDPRank scores and the
baselines provided for the CORA dataset is done.
<br>
<br>
<br>
<img src = "https://github.com/sans-sehgal/Information-Retrieval/blob/main/Images/final%20comparison.png"></img>
<br>

</div>

<h4> For a more detailed analysis please check out this <a href="https://drive.google.com/file/d/1mQ564F_1NBC1EWwx3hb0Gn0HKfciByOS/view"> Technical Report </a>. </h4>
