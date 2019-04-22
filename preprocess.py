import numpy as np 
import os 
import time
import pickle
from tqdm import tqdm

max_features = 12000
ft = 'data/wiki-news-300d-1M.vec'

with open('data/categories.pkl', 'rb') as f:
    cats = pickle.load(f)
word_ix = {word: ix for ix, word in enumerate(cats)}

def load_embeddings(word_ix: dict, save_outputs=False):
    r"""
    Function for preparing the raw fasttext embedding files and optionally saving them as processed pickle files
    -----------------------------------------
    Arguments:

    word_ix (dict): Dictionary of Quick, Draw! class strings to be processed and their respective indices
    save_outputs (boolean): Whether to save the processed files as pickle files

    """
    load_start = time.time()
    EMBEDDING_FILE = ft
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in tqdm(open(EMBEDDING_FILE)) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_ix))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in tqdm(word_ix.items()):
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    print(f"Loaded fasttext vectors in {(time.time() - load_start) / 60  :.2f} minutes.")

    if save_outputs:
        for filename in ['embed_matrix', 'embed_ix']:
            with open(f'data/{filename}.pkl', 'wb') as f:
                pickle.dump(filename, f)
        print(f'Outputs saved to pickle files')
    
    return embedding_matrix, embeddings_index

