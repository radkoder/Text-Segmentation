NOT_FOUND_MSG = '''
    Small polish segmentation dataset not found...
    This is private dataset and is not released to the public for now.
    If you have this dataset installed - make sure the file:
        * 23_Wydobywanie_struktury.zip
    Is located in the /data folder in the root dir 
    '''
from datasets import DATASET_DIR, TMP_DIR
from common import files, stage, utils
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
import os, shutil, re, io
from collections import deque
from models import sentence_bert as sbert

def get(variant = None):
    if not check():
        return None
    else:
        return os.path.abspath(os.path.join(DATASET_DIR,'Wydobywanie_struktury.zip'))
    
def check():
    sset =  os.path.abspath(os.path.join(DATASET_DIR,'Wydobywanie_struktury.zip'))
    s_found = os.path.isfile(sset)
    print(f'{sset} found: {s_found}')
    if not (s_found):
        print(NOT_FOUND_MSG)
    return s_found


def parse_par(parline):
    Ls = parline.replace('. ','.\n').splitlines()[:-1]
    i = 1 
    while i < len(Ls):
        if not Ls[i][0].isupper():
            Ls[i-1] = ' '.join([Ls[i-1],Ls[i]])
            Ls.pop(i)
        else:
            i+=1
    return Ls
#parse_par(raw.readlines()[2])
def split_pars(rawlines):
    rawlines = deque(rawlines)
    frag_start = re.compile("\[##Fragment_\d+##\]")
    frag_end = re.compile("\[@@Fragment_\d+@@\]")
    title = rawlines.popleft().strip() if not frag_start.match(rawlines[0]) else None
    curr_par_n= 0
    curr_par = [] # num, title, content
    pars = []
    for l in rawlines:
        if frag_start.match(l):
            curr_par_n +=1
            curr_par.append(curr_par_n)
        elif frag_end.match(l):
            if len(curr_par) == 1:
                print(f"Fragment {curr_par_n} is empty")
            elif len(curr_par) == 2:
                pars.append(tuple([curr_par[0],None,curr_par[-1]]))
                curr_par.clear()
            else:  
                pars.append(tuple([curr_par[0],curr_par[1],' '.join(curr_par[2:])]))
                curr_par.clear()
        else:
            curr_par.append(l.strip())
    return pars
        
def parse_doc(reader):
    out = io.StringIO()
    lines = [l.decode('utf-8') for l in reader.readlines()] # decode lines
    for n,tit,con in split_pars(lines):
        out.write(f'=========={n},{tit}============\n')
        out.write('\n'.join(parse_par(con))+'\n')
    return out
def get_sentences(infile,name):
    with files.zipped(infile) as wiki_ds:
        reader = files.open_zipped(wiki_ds,name)
        parsed = parse_doc(reader)
        lines = parsed.getvalue().splitlines() # decode lines
    return lines

@stage.measure("Making sentence embeddings")
def make_embeddings(infile, outfile, embedder = 'trans', trunc = None):
    sentence_bert = sbert.get(embedder)
    with files.zipped(infile) as wiki_ds:
        if trunc == None:
            members = [m for m in files.zipped_members(wiki_ds) if files.is_zipped_file(m)]
        else:
            members = [m for m in files.zipped_members(wiki_ds)[:trunc] if files.is_zipped_file(m)]
        bar = stage.ProgressBar("Embedding sentences",len(members))
        for name in members:
            bar.update(files.zipped_name(name))
            reader = files.open_zipped(wiki_ds,name)
            parsed = parse_doc(reader)
            lines = parsed.getvalue().splitlines() # decode lines
            lines = [l for l in lines if not l.startswith('***')] # filter special makers
            sentences = [l for l in lines if not l.startswith('===')]
            div_i = np.array([i for i,line in enumerate(lines) if line.startswith('===')] + [len(lines)])
            div_t = np.repeat(np.arange(len(div_i)-1),np.diff(div_i)-1)
            embeddings = sentence_bert(sentences)
            assert len(div_t) == len(embeddings), f'{len(div_t)} != {len(embeddings)}'
            np.save(files.file(TMP_DIR+'/'+files.zipped_name(name)+'_seg.npy','wb'),div_t)
            np.save(files.file(TMP_DIR+'/'+files.zipped_name(name)+'_emb.npy','wb'),embeddings)
            #print(f'{name.name} : {div_i} : {len(sentences)} = {len(div_t)} : {embeddings.shape}')
        bar.end()
    shutil.make_archive(outfile,'zip',root_dir=TMP_DIR,base_dir='.')
    os.replace(outfile+'.zip',outfile+'.npz')
    shutil.rmtree(TMP_DIR)