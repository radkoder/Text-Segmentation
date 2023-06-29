import numpy as np
from scipy.cluster import vq
def encode_blocks(v, mode = 'max'):
    if mode == 'max':
        return np.max(v,2)
    elif mode == 'v-mean':
        return np.array([u/np.sqrt(u.dot(u.T)) for u in np.sum(v,2)])
    elif mode == 'sum':
        return np.sum(v,2)
    elif mode == 'mean':
        return np.mean(v,2)
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
        return -np.sum(np.abs(v1-v2))
    elif mode == 'diff-len':
        u = v1-v2
        return np.sqrt(np.dot(u,u))
def sub_mean(Vs):
    s = np.sum(Vs,0)
    s /= np.sqrt(s.dot(s.T))
    return Vs - s
def check_min(arr, idx,radius=1):
    lower = max(idx-radius,0)
    upper = min(idx+radius+1,len(arr))
    return arr[idx] == min(arr[lower:upper])

def main(emb,threshold = None, block_size = None, block_mode = 'max', whiten = False, cmp_mode ='dot', substract_mean = False, std_cutoff = 1.0, check_minimum = True, check_radius=1):
    if block_size is None: BS = 3
    else : BS = min(block_size,len(emb)//3)

    v = vq.whiten(emb,check_finite=False) if whiten else emb
    v = sub_mean(v) if substract_mean else v
    v = np.lib.stride_tricks.sliding_window_view(v,BS,0)
    v = encode_blocks(v,block_mode)
    
    score = np.array([similarity_measure(v[i],v[i+BS],cmp_mode) for i in range(len(v)-BS)])
    t = np.mean(score) - std_cutoff*np.std(score) if threshold is None else threshold
    divs_i =  np.array([m for m in range(len(score)) if score[m] < t and (check_min(score,m,check_radius) if check_minimum else True)]) + (BS-1)
    print(divs_i)
    if len(divs_i) < 3:
        '''Worst case - just take a minimal three'''
        divs_i = np.argsort(score)[:4-len(divs_i)] + (BS-1)
    print(divs_i)
    return np.array([np.eye(1,len(emb),i) for i in divs_i],dtype=int).sum(0)[0]


def with_params(**kwargs):
    return lambda x: main(x,**kwargs)