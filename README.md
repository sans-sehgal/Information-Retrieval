# Information-Retrieval

### ABSTRACT 
<p style="text-align: justift;">
  
Finding pertinent information from a variety of sources is known as information retrieval (IR). Since
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
</p>

### METHODOLOGY

We use RL algorithms for ranking while utilising Graph Neutral Networks or GNN’s to generate node
embeddings. The first step in the process is calculating the node embeddings of documents. Libraries
such as Node2Vec and DeepWalk are used to calculate embeddings of the nodes. In our problem
statement, the nodes are the documents that need to be ranked and the edges are the links between the
documents. For instance, in the scenario of searching for research papers, the nodes will be the papers
themselves and the links will be the citations these papers carry to different papers within the dataset.


![Node Embeddings](https://github.com/sans-sehgal/Information-Retrieval/blob/main/Images/Node%20Embeddings.png)
