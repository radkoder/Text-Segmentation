import numpy as np

def array_to_nltk(seg_array):
    return ''.join(seg_array.astype('str'))

def tags_to_array(seg_tags):
    seg_true = np.pad(np.diff(seg_tags),(0,1))
    return np.array([1 if s > 0 else 0 for s in seg_true])
def array_to_tags(seg_array):
    ts = []
    i = 0
    for a in seg_array:
        ts.append(i)
        if a > 0: i+=1
    return ts
def count(seg_tags):
    return len(np.unique(seg_tags))