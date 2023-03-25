import numpy as np
def sim_score(v):
    return np.array([v[i].dot(v[i+1]) for i in range(len(v)-1)])
def depth_score(v):
    return np.array([((np.max(v[:w]) + np.max(v[w:]))/2) - v[w] for w in range(1,len(v)-1)])
def local_minima_i(v):
    return np.array([i for i in range(1,len(v)-1) if (v[i-1] >= v[i] and v[i+1] >= v[i]) ])

def simple_seg(emb):
    '''
    This function performs naive segmentation based on similarity scores between embeddings
    Process is similar to the one described in TopicTiling / TextTiling
    
    It's pretty bad - mainly used as reference
    '''
    ss = sim_score(emb)
    lm = local_minima_i(ss)
    dp = depth_score(ss)
    divs_i =  np.array([m for m in lm if m < len(dp) and dp[m-1] >= (np.mean(dp) - np.std(dp)) ])
    if len(divs_i) == 0:
        '''Worst case - just take a minimum'''
        divs_i = np.array([np.argmin(ss)])
        print(f' argmin: {divs_i}')
    return np.array([np.eye(1,len(emb),i) for i in divs_i],dtype=int).sum(0)[0]



