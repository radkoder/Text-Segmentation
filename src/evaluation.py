from nltk.metrics.segmentation import pk, windowdiff
from datasets import EmbeddedDataset
from algorithms import segments
import numpy as np
from common import stage

def evaluate_document(true,infered,k):
    ref = segments.array_to_nltk(true)
    hyp = segments.array_to_nltk(infered)
    return pk(ref,hyp,k) ,windowdiff(ref,hyp,k)

def evaluate_segmenter(dataset : EmbeddedDataset, seg_alg, k = None, silent = False):
    Ps,Ws = [],[]
    if not silent: bar = stage.ProgressBar("Evaluating",len(dataset.get_embeddings()))
    for emb in dataset.get_embeddings(): 
        if not silent: bar.update(emb)
        seg =  dataset.segmentation(emb)
        inferred = seg_alg(dataset[emb])
        seg_true = segments.tags_to_array(dataset[seg])
        p,w = evaluate_document(seg_true,inferred,k)
        Ps.append(p)
        Ws.append(w)
    if not silent: bar.end()
    return Ps, Ws
