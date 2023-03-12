NOT_FOUND_MSG = '''
    Wiki_727K dataset not found!
    Make sure the files:
    * wiki_727K.tar.bz2
    * wiki_test_50.tar.bz2

    Are placed in the /data folder in root directory
    The download link used to aquire the dataset is: (may not be working at the time of reading)
    https://www.dropbox.com/sh/k3jh0fjbyr0gw0a/AADzAd9SDTrBnvs1qLCJY5cza?dl=0
    '''
from datasets import DATASET_DIR, TMP_DIR
from common import files, stage
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
import os, shutil

def get(variant = None):
    if not check():
        return None
    elif variant == 'full':
        return os.path.abspath(os.path.join(DATASET_DIR,'wiki_727K.tar.bz2'))
    else: return os.path.abspath( os.path.join(DATASET_DIR,'wiki_test_50.tar.bz2'))

def check():
    bigdataset = os.path.abspath(os.path.join(DATASET_DIR,'wiki_727K.tar.bz2'))
    smallset =os.path.abspath( os.path.join(DATASET_DIR,'wiki_test_50.tar.bz2'))

    b_found = os.path.isfile(bigdataset)
    s_found = os.path.isfile(smallset)

    print(f'{bigdataset} found: {b_found}')
    print(f'{smallset} found: {s_found}')
    if not (b_found and s_found):
        print(NOT_FOUND_MSG)
    return b_found or s_found


@stage.measure("Making sentence embeddings")
def make_embeddings(infile, outfile):
    print('Loading sentence bert')
    sentence_bert = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    with files.zipped(infile) as wiki_ds:
        members = [m for m in wiki_ds.getmembers() if m.isfile()]
        bar = stage.ProgressBar("Embedding sentences",len(members))
        for name in members:
            bar.update(name.name)
            reader = wiki_ds.extractfile(name)
            lines = [l.decode('utf-8') for l in reader.readlines()] # decode lines
            lines = [l for l in lines if not l.startswith('***')] # filter special makers
            sentences = [l for l in lines if not l.startswith('===')]
            div_i = np.array([i for i,line in enumerate(lines) if line.startswith('===')] + [len(lines)])
            div_t = np.repeat(np.arange(len(div_i)-1),np.diff(div_i)-1)
            embeddings = sentence_bert(sentences)
            assert len(div_t) == len(embeddings), f'{len(div_t)} != {len(embeddings)}'
            np.save(files.file(TMP_DIR+'/'+name.name+'_seg.npy','wb'),div_t)
            np.save(files.file(TMP_DIR+'/'+name.name+'_emb.npy','wb'),embeddings)
            #print(f'{name.name} : {div_i} : {len(sentences)} = {len(div_t)} : {embeddings.shape}')
        bar.end()
    shutil.make_archive(outfile,'zip',root_dir=TMP_DIR,base_dir='.')
    os.replace(outfile+'.zip',outfile+'.npz')
    shutil.rmtree(TMP_DIR)


    


