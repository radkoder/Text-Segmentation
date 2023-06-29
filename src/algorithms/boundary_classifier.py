import tensorflow as tf
import numpy as np
from datasets import windowed

def main(emb, classifier : tf.keras.Model, context_size = 5, threshold = 0.5):
    probs = []
    for n in range(emb.shape[0]):
        window = windowed.padded_window(emb,n,context_size)[np.newaxis]
        probs.append(np.squeeze(classifier(window).numpy()))
    return np.array([1 if p > threshold else 0 for p in probs])

def with_params(**kwargs):
    return lambda x: main(x,**kwargs)

