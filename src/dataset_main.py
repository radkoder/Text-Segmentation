import sys
from datasets import *
HELP_STRING = '''
Usage:
python dataset_main.py VERB DATASET
Where:
    VERB = prepare
    DATASET = wiki | wiki_test
'''
def main():
    if sys.argc > 3:
        verb = sys.argv[1]
        dataset = sys.argv[2]
    else:
        print(HELP_STRING)
        exit(0)
    if verb == 'prepare':
        if dataset == 'wiki':
            wikiset.make_embeddings(wikiset.get('full'),'wiki.npz')
        elif dataset == 'wiki_test':
            wikiset.make_embeddings(wikiset.get(),'wiki_test.npz')
        else: 
            print(f"Unknown dataset: {dataset}")
            exit(-1)

if __name__ == '__main__':
    main()