from nltk.metrics.segmentation import pk, windowdiff
from datasets import EmbeddedDataset
import segments
import numpy as np

def evaluate_document(true,infered,k):
    ref = segments.array_to_nltk(true)
    hyp = segments.array_to_nltk(infered)
    return pk(ref,hyp,k) ,windowdiff(ref,hyp,k)

def evaluate_segmenter(dataset : EmbeddedDataset, seg_alg, k = None):
    Ps,Ws = [],[]
    for emb in dataset.get_embeddings(): 
            seg =  dataset.segmentation(emb)
            inferred = seg_alg(dataset[emb])
            seg_true = segments.tags_to_array(dataset[seg])
            p,w = evaluate_document(seg_true,inferred,k)
            Ps.append(p)
            Ws.append(w)
    print(f'Pmean: {np.mean(Ps)} Pstv: {np.std(Ps)}')
    print(f'Wmean: {np.mean(Ws)} Wstv: {np.std(Ws)}')
    return Ps, Ws
