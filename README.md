# Text-Segmentation
Collection of Text Segmentation algorithms, collected for use in my master thesis - _"Identifying thematic sections in documents based on text embedding models"_ (2023).
The models here implemented rely on sentence vector embeddings. In the code [this google provided model](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3) is used although any sentence-wise vector embedding method can be substituted.

## Implemented algorithms
The implementations of following algorithms can be found in this repo:
1. *Simple* (`algorithms.simple`)  
Baseline algorithm based on Hearst (1997) paper about segmentation. It's performance is pretty bad, serving only as a reference for other models.
2. *GraphSeg* (`algorithms.graphseg`)  
Implementation of the GraphSeg algorithm from (Glava et al. 2016) paper. One diffrence being that instead of using word embeddings to produce sentence representations, the sentence encoder is used. Running of this algorithm requires finding maximal cliques in a graph which is an NP problem - the runtime of this algorithm can be horrendously slow given some parameters.
3. *Block Comparison* (`algorithms.block_comparison`)  
Implementation of algorithm described in (Solbiati et al. 2021), it is an evolution of hearst original algorithm operating on vector sentence embeddings.
This is the most parametrized model in the collection - Block encoding, Block comparison, standarization etc. are modifiable by  hiperparameters.
4. *Boundary Classifier* (`algorithms.boundary_classifier`)
Family of algorithms that, given a context of senctences around the point, is trying to predict the probability of boundary at that point. This approach is well suited for use of supervised learning algorithms, as seen in (Omri et al. 2018) which uses LSTMs to estimate the probabilities of boundaries.
