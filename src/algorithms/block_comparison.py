import numpy as np
from scipy.cluster import vq
def encode_blocks(v, mode = 'max'):
    if mode == 'max':
        return np.max(v,2)
    elif mode == 'mean':
        return np.array([u/np.sqrt(u.dot(u.T)) for u in np.sum(v,2)])
    elif mode == 'sum':
        return np.sum(v,2)
    else:
        print(f"Unknown mode: {mode}")
        return None
def similarity_measure(v1,v2, mode = 'dot'):
    if mode == 'dot':
        return v1.dot(v2)
    elif mode == 'cos':
        return v1.dot(v2)/np.sqrt(v1.dot(v1) * v2.dot(v2))
    elif mode == 'acos':
        # linear similarity
        return np.pi - np.arccos(v1.dot(v2)/np.sqrt(v1.dot(v1) * v2.dot(v2)))
    elif mode == 'max-abs-diff':
        return -np.max(np.abs(v1-v2))
    elif mode == 'sum-diff':
        return -np.max(np.abs(v1-v2))
def sub_mean(Vs):
    s = np.sum(Vs,0)
    s /= np.sqrt(s.dot(s.T))
    return Vs - s

def main(emb,threshold = None, block_size = None, block_mode = 'max', whiten = False, cmp_mode ='dot', substract_mean = False):
    if block_size is None: BS = 3
    else : BS = min(block_size,len(emb)//3)

    v = vq.whiten(emb,check_finite=False) if whiten else emb
    v = sub_mean(v) if substract_mean else v
    v = np.lib.stride_tricks.sliding_window_view(v,BS,0)
    v = encode_blocks(v,block_mode)
    
    score = np.array([similarity_measure(v[i],v[i+BS],cmp_mode) for i in range(len(v)-BS)])
    t = np.mean(score) -np.std(score) if threshold is None else threshold
    divs_i =  np.array([m for m in range(len(score)) if score[m] < t]) + (BS-1)
    if len(divs_i) == 0:
        '''Worst case - just take a minimum'''
        divs_i = np.argsort(score)[:3] + (BS-1)
    return np.array([np.eye(1,len(emb),i) for i in divs_i],dtype=int).sum(0)[0]


def with_params(**kwargs):
    return lambda x: main(x,**kwargs)