NOT_FOUND_MSG = '''
    Choi dataset not found!
    Make sure the file:
    * Choi-3-11.zip

    Is placed in the /data folder in root directory
    The download link used to aquire the dataset is: (may not be working at the time of reading)
    https://github.com/logological/C99/tree/master/data/3/3-11
    '''
from datasets import DATASET_DIR, TMP_DIR
from common import files, stage
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
import os, shutil
from models import sentence_bert as sbert

def get(variant = None):
    if not check():
        return None
    else:
        return os.path.abspath(os.path.join(DATASET_DIR,'Choi-3-11.zip'))
   

def check():
    ds = os.path.abspath(os.path.join(DATASET_DIR,'Choi-3-11.zip'))
    b_found = os.path.isfile(ds)
    print(f'{ds} found: {b_found}')
    if not b_found:
        print(NOT_FOUND_MSG)
    return b_found

def get_raw_lines(dsfile, filename):
    with files.zipped(dsfile) as wiki_ds:
        members = [m for m in wiki_ds.infolist() if not m.is_dir()]
        print(members)
        if filename not in members:
            print(f'File {filename} not found in archive {dsfile}')
            return None
        else:
            with dsfile.open(filename) as reader:
                lines = [l.decode('utf-8') for l in reader.readlines()] # decode lines
            return lines

def get_unsegmented_lines(dsfile, filename):
    return [l for l in get_raw_lines(dsfile,filename) if not l.startswith('===') and not l.startswith('***')]


@stage.measure("Making sentence embeddings")
def make_embeddings(infile, outfile, embedder = 'trans'):
    sentence_bert = sbert.get(embedder)
    with files.zipped(infile) as wiki_ds:
        members = [m for m in wiki_ds.infolist() if not m.is_dir()]
        bar = stage.ProgressBar("Embedding sentences",len(members))
        for name in members:
            bar.update(name.filename)
            with wiki_ds.open(name) as reader:
                lines = [l.decode('utf-8') for l in reader.readlines()] # decode lines
            lines = [l for l in lines if not l.startswith('***')] # filter special makers
            sentences = [l for l in lines if not l.startswith('===')]
            div_i = np.array([i for i,line in enumerate(lines) if line.startswith('===')] + [len(lines)])
            div_t = np.repeat(np.arange(len(div_i)-1),np.diff(div_i)-1)
            embeddings = sentence_bert(sentences)
            assert len(div_t) == len(embeddings), f'{len(div_t)} != {len(embeddings)}'
            np.save(files.file(TMP_DIR+'/'+name.filename+'_seg.npy','wb'),div_t)
            np.save(files.file(TMP_DIR+'/'+name.filename+'_emb.npy','wb'),embeddings)
            #print(f'{name.name} : {div_i} : {len(sentences)} = {len(div_t)} : {embeddings.shape}')
        bar.end()
    shutil.make_archive(outfile,'zip',root_dir=TMP_DIR,base_dir='.')
    os.replace(outfile+'.zip',outfile+'.npz')
    shutil.rmtree(TMP_DIR)