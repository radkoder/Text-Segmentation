
from networkx.algorithms import approximation
from networkx.algorithms import clique
import networkx as nx
from . import segments
import numpy as np

def sg(segs,i):
    for s in range(len(segs)):
        if i in segs[s]: return s
    else:
        return -1
def sgr(seg1,seg2,emb_mat):
    return np.mean([emb_mat[s1,s2] for s1 in seg1 for s2 in seg2])
def merged_seg(segs, i, j):
    s1 = segs.pop(max(i,j))
    s2 = segs.pop(min(i,j))
    segs.append(s2+s1)
    return sorted(segs)
def merge_small_seg(segs,emb_mat,min_n):
    for i in range(len(segs)):
        if len(segs[i]) < min_n:
            if i == (len(segs)-1):
                return merged_seg(segs,i-1,i)
            elif i == 0:
                return merged_seg(segs,i,i+1)
            elif sgr(segs[i],segs[i+1],emb_mat) > sgr(segs[i],segs[i-1],emb_mat):
                return merged_seg(segs,i,i+1)
            else:
                return merged_seg(segs,i-1,i)

def sbert_graphseg(emb,threshold = 0.5,n_min_seg = 3, clique_finder = 'precise'):
    mat = np.inner(emb,emb)
    adj = np.array(mat > threshold,dtype=int)
    G = nx.Graph(adj)
    if clique_finder == 'precise':
        cliques  = clique.find_cliques(G)
    elif clique_finder == 'clique_removal':
        _,cliques =  approximation.clique.clique_removal(G)
    
    cliques = list(cliques)
    segs = []

    #segments from cliques
    for q in cliques:
        for si,sj in ((i,j) for i in q for j in q):
            if sj - si != 1: continue
            if sg(segs,si) < 0 and sg(segs,sj) < 0:
                segs.append([si,sj])
            elif sg(segs,si) >= 0 and sg(segs,sj) < 0:
                segs[sg(segs,si)].append(sj)
            elif sg(segs,si) < 0 and sg(segs,sj) >= 0:
                segs[sg(segs,sj)].append(si)

    # fill orphaned sentences
    for i in range(len(emb)):
        if sg(segs,i) < 0:
            segs.append([i])
    segs = sorted(segs)

    # Merge similar segments
    for q in cliques:
        for si,sj in ((i,j) for i in q for j in q):
            if sg(segs,si) - sg(segs,sj) == 1:
                segs = merged_seg(segs,sg(segs,si),sg(segs,sj))

    #Merge small segments
    num_small_segs = lambda s: sum([1 if len(es) < n_min_seg else 0 for es in s])
    while num_small_segs(segs) > 0:
        segs = merge_small_seg(segs,mat,n_min_seg)

    #to tags:
    seg_tags = []
    for i in range(len(segs)):
        seg_tags += [i]*len(segs[i])
    return segments.tags_to_array(seg_tags)

def with_params(**kwargs):
    return lambda x : sbert_graphseg(x,**kwargs)