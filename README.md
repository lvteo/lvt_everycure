# Li-Vern Teo Every Cure Submission 

This repo contains the Data Science Assignment submission for EveryCure. Please follow the instructions of this README to replicate results

## Set-Up 

This project was completed with Python 3.7.17 due to package dependencies. All packages with versions are stored in the `requirements.txt` file and can be installed in a virtual environment with 

```
python3.7 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Files Descriptions 

| File | Description |
|------|-------------|
| input/Edges.csv| Provided Edges document (please upload)  |
| input/Ground Truth.csv     |    Provided Ground Truth Document  (please upload)       |
| input/Nodes.csv     |     Provided Nodes document (please upload)       |
| processed/ground_split.csv     | Ground Truth csv, but with test, train and validation assignments. See train_test_val_split.ipynb for methodology             |
| processed/baseline     | Nodes and Edges used for predicting links from the baseline model, with features generated with Bag of Words             |
| processed/experiment     | Nodes and Edges used for predicting links from the experiment model, with features generated using BioBert embeddings             |
| eda.ipynb     | Jupyter notebook with quick eda of nodes and edges             |
| train_test_val_split.ipynb     | Jupyter notebook to crease processed/ground_split.csv             |
| baseline_bow.ipynb     | Jupyter notebook with preprocessing, training and evaluation of baseline model             |
| biobert_embeddings.ipynb     | Jupyter notebook with preprocessing, training and evaluation of experiment model             |


## Approach 

At a high level, I approached the problem by using NLP techniques to generate features for nodes, and using Link Prediction to classify drug disease pairs. I did this by: 

1. Establish a baseline using Bag of Words as a naive technique of obtaining node features 
2. Experiment with using BioBert to get sentence embeddings as node features

### Decisions 

#### 1. Using Deep Graph Library (DGL) for Link Prediction 

Some research led me to the DGL documentation https://docs.dgl.ai/ which I found to be well-written and easily understandable. I ultimately chose this package due to its guided notebooks, despite some challenges around torch dependencies. DGL provides the GraphSAGE implementation, which has improvements over a standard Graph Convolution Network. Mainly it utilises the SEAL framework (https://papers.nips.cc/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf) which samples neighbourhoods and subgraphs rather than all nodes in the graph. In theory, this is computationally more efficient, and should make the graph more robust to unseen nodes during prediction.

#### 2. Reducing the Problem Statement to only Nodes in Ground Truth

In both the baseline and experiment, the problem statement is reduced to only the nodes and edges in the ground truth. This was ultimately due to computing restraints, and a need to work quickly through the problem. The ground truth provided about 28k edges and 3k nodes. Given the the max dimensionality in both training runs was 900, I was satisfied that there were enough samples despite the sparse features.

#### 3. Baseline with Bag of Words 

Bag of Words is a naive approach that simply counts the number of word occurrences in each sample based on the corpus. I thought this to be an adequate method for feature generation as it is quick and easy to implement.


#### 4. Using BioBert for embeddings 

BioBERT is a transformer pre-trained on a biomedical corpus. It is free to utilise from HuggingFace and in theory would provide more nuance and rich features compared to Bag of Words, as such it was chosen as a contender. In my research, I found BioGPT which purports to have improvements over BioBERT. Based on my readings however, it seems to be optimised for other uses such as text generation or summarisation. I ultimately chose BioBERT as we're only concerned with embeddings at this stage. 


#### 5. Adding the node's category as a feature 

This was done by concating the NLP based features with a simple label encoding of the node's categories. I thought this information would be significant to understanding the types of nodes that can be connected at all, and the potential direction of an edge. 


#### 6. Creating a new feature = node name + node description 

Concating the node name and node description provides more information during feature generation, also it serves as a imputation technique as some nodes only had one or the other. 

#### 7. Reducing the length of node descriptions for word embeddings 

This was due to limitations in the context window size. Prior to limiting the character input, I attempted a technique of splitting the full descriptions into chucks, getting embeddings of each chunk, and taking the average as the full embedding. This is due to computational limitation as my kernel would periodically die during this process.


## Conclusions and Findings

Model performance was evaluated by analysing the training and validation loss charts, and the ROC-AUC curve. The ROC-AUC curve is well suited to this task as it is a binary prediction, and classes are relatively balanced. 

Ultimately, the baseline model outperformed the experimental one. This might be due to computational limitation of generating embeddings for the node descriptions. While BoW is a more naive approach, its quick implementation allowed me to utlise all the information in the description, while the BioBERT implementation was limited to the first 50 characters. 


## Room for Improvement

#### 1. Improvements to model architecture 

The GraphSAGE model is a very simple 2 layer graph. Plots of train and validation loss in the baseline model show some sign of overfitting. This architecture could benefit from some regularisation like drop-out layers, early stopping, l1/2 regularisation etc


#### 2. Modularisation of Code 

Since the training, evaluation and some preprocessing is repeated across the baseline and experiment, the code can be cleaned up with more modularisation.


#### 3 Edge features

There was room to incorporate more information about the edges, which can be stored and utilised in the same manner as the nodes. However, this was dropped due to time limitation. 
