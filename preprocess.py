import numpy as np 
import os 
import time
import pickle
import subprocess
from tqdm import tqdm

def _process_embeddings(embedding_file, word_ix, save_outputs=True):
    """
    Function for preparing the raw fasttext embedding files and saving them as processed pickle files

    Arguments:
    embedding_file (str): Path to the raw embedding file
    word_ix (dict): Dictionary of Quick, Draw! class strings to be processed and their respective indices
    save_outputs (boolean): Whether to save the processed files as pickle files

    """
    load_start = time.time()
    max_features = 12000
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in tqdm(open(embedding_file)) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_ix))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in tqdm(word_ix.items()):
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    print(f"Loaded and processed fasttext vectors in {(time.time() - load_start) / 60  :.2f} minutes.")

    if save_outputs:
        with open(f'data/embed_matrix.pkl', 'wb') as f:
            pickle.dump(embedding_matrix, f)
        with open(f'data/embed_ix.pkl', 'wb') as f:
            pickle.dump(embeddings_index, f)
        print(f'Outputs saved to pickle files.')
    


if __name__ == "__main__":

    ft = 'data/wiki-news-300d-1M.vec'
    subprocess.run(['wget', '-O', ft+'.zip', 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip'])
    subprocess.run(['unzip', ft+'.zip'])
    subprocess.run(['rm', ft+'.zip'])
    with open('data/categories.pkl', 'rb') as f:
        cats = pickle.load(f)
    word_ix = {word: ix for ix, word in enumerate(cats)}
    _process_embeddings(ft, word_ix, save_outputs=True)
