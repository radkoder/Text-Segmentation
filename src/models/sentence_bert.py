
from common import stage
import tensorflow_hub as hub
import tensorflow_text

_cnn_model = None
_transformer_model = None

@stage.measure("Loading sentence bert")
def get(type):
    global _cnn_model
    global _transformer_model
    ''' Get one of 'cnn' or 'trans' sentence encoder models'''
    if type == 'cnn':
        if _cnn_model is None:
            _cnn_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        return _cnn_model
    elif type == 'trans':
        if _transformer_model is None:
            _transformer_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
        return _transformer_model
    else:
        print(f"Unknown model: {type}")
        return None

