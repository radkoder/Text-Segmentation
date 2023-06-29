import os
DATASET_DIR = os.path.abspath('../data')
TMP_DIR = DATASET_DIR+'/tmp'
__all__ = ["wikiset"]
import numpy as np
from common import utils


class EmbeddedDataset(object):
    def __init__(self, filename):
        self.filename = filename
    def __enter__(self):
        self.file = np.load(self.filename, mmap_mode='r')
        return self
    def __exit__(self,exc_type, exc_value, traceback):
        self.file.close()
    def __getitem__(self, key) -> np.ndarray:
        if isinstance(key, str):
            return self.file[key]
        elif utils.has_method(key,'__getitem__'):
            return [self.file[k] for k in key]
        else:
            return None
    def __len__(self):
        return len(self.get_embeddings())
    def open(self):
        self.file = np.load(self.filename, mmap_mode='r')
    def close(self):
        self.file.close()
    def get_members(self):
        return self.file.files
    def document_lengths(self):
        return np.array([len(self.file[m]) for m in self.get_segments()])
    def segment_lengths(self):
        return np.array([len(np.unique(self.file[m])) for m in self.get_members() if m.endswith('_seg')])

    def get_segments(self):
        return np.array([m for m in self.get_members() if m.endswith('_seg')])

    def get_embeddings(self):
        return np.array([m for m in self.get_members() if m.endswith('_emb')])
    
    def embbeding(self,seg_name):
        return seg_name[:-4]+'_emb'

    def segmentation(self,emb_name):
        return emb_name[:-4]+'_seg'

    def get_file(self, name):
        return self.file[name+'_seg'], self.file[name+'_emb']
    def mean_segment_length(self):
        return np.mean(self.document_lengths()/self.segment_lengths(),dtype=int)
            