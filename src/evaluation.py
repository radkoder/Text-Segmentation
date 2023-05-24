from nltk.metrics.segmentation import pk, windowdiff
from datasets import EmbeddedDataset
from algorithms import segments
import numpy as np
from common import stage

def evaluate_document(true,infered,k, wd_weighted = False):
    ref = segments.array_to_nltk(true)
    hyp = segments.array_to_nltk(infered)
    return pk(ref,hyp,k) ,windowdiff(ref,hyp,k, weighted=wd_weighted)

def get_n_seg_diff(true,inferred):
    ref = segments.array_to_nltk(true).count('1')
    hyp = segments.array_to_nltk(inferred).count('1')
    return hyp - ref

def evaluate_segmenter(dataset : EmbeddedDataset, seg_alg, k = None, silent = False, wd_weighted = False, seg_diff= False):
    Ps,Ws = [],[]
    Ds = []
    if not silent: bar = stage.ProgressBar("Evaluating",len(dataset.get_embeddings()))
    for emb in dataset.get_embeddings(): 
        if not silent: bar.update(emb)
        seg =  dataset.segmentation(emb)
        inferred = seg_alg(dataset[emb])
        seg_true = segments.tags_to_array(dataset[seg])
        if seg_diff:
            Ds.append(get_n_seg_diff(seg_true,inferred))
        p,w = evaluate_document(seg_true,inferred,k, wd_weighted = wd_weighted)
        Ps.append(p)
        Ws.append(w)
    if not silent: bar.end()
    if seg_diff:
        return Ps, Ws, Ds
    else:
        return Ps, Ws
