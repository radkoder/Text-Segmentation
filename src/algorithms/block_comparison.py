import numpy as np
def main(emb,threshold = None, block_size = None):
    if block_size is None: BS = 3
    else : BS = block_size

    v = np.lib.stride_tricks.sliding_window_view(emb,BS,0)
    v = np.max(v,2)
    score = np.array([v[i].dot(v[i+BS]) for i in range(len(v)-BS)])
    t = np.mean(score) -np.std(score) if threshold is None else threshold
    divs_i =  np.array([m for m in range(len(score)) if score[m] < t]) + (BS-1)
    if len(divs_i) == 0:
        '''Worst case - just take a minimum'''
        divs_i = np.argsort(score)[:3] + (BS-1)
    return np.array([np.eye(1,len(emb),i) for i in divs_i],dtype=int).sum(0)[0]

def with_params(threshold=None,block_size=None):
    return lambda x: main(x,threshold,block_size)