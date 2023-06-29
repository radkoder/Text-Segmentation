import numpy as np
import tensorflow as tf
import datasets
from algorithms import segments

def padded_window(v,n,size):
    if n > size and n < v.shape[0]-size:
        return v[n-size:n+size]
    else:
        low_bound = n - size
        high_bound = n + size
        c = v[max(low_bound,0):min(high_bound,v.shape[0])]
        low_pad = max(-low_bound,0)
        high_pad = max(0,high_bound-v.shape[0])
        return np.pad(c,((low_pad,high_pad),(0,0)))

def make_padded_windows(embs,segs,window_size):
    Embs = np.vstack([np.array([padded_window(e,n,window_size) for n in range(e.shape[0])]) for e in embs])
    Segs = np.concatenate([np.array([seg[n] for n in range(seg.shape[0])]) for seg in segs])
   
    return Embs, Segs

class WindowedDataset(tf.keras.utils.Sequence):
    def __init__(self,emb_dataset : datasets.EmbeddedDataset, batch_size = 8, weighted = True, limits = (0,None)):
        self.ds = emb_dataset
        self.weighted = weighted
        self.emb_files = self.ds.get_embeddings()[slice(*limits)]
        self.seg_files = [self.ds.segmentation(ef) for ef in self.emb_files]
        self.length = len(self.emb_files)
        self.batch_size = batch_size
    def __len__(self):
        return self.length // self.batch_size
    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size,self.length)
        emb_batch = self.ds[self.emb_files[low:high]]
        seg_batch = [segments.tags_to_array(s) for s in self.ds[self.seg_files[low:high]]]
        Xs, Ys = make_padded_windows(emb_batch,seg_batch,5)
        if self.weighted:
            weights = Ys*8 + np.ones_like(Ys)
            return Xs, Ys, weights
        else:
            return Xs,Ys
