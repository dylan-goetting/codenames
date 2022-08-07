import argparse
import gensim
import gensim.downloader
import pickle
import gensim.models
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import re

def filter_word(word, pos, neg):
    if re.match(r'[^a-z]', word):
        return False
    if len(word) <= 1:
        return False
    if word in pos:
        return False
    if word in neg:
        return False
    return True

def serialize_models():

    model_names = ['fasttext-wiki-news-subwords-300', 'glove-twitter-100', 
    'glove-wiki-gigaword-100', 'word2vec-google-news-300', 'word2vec-ruscorpora-300']

    for mod in model_names:
        model = gensim.downloader.load(mod)
        valid_words = []
        for i in range(len(model.index_to_key)):
            if filter_word(model.index_to_key[i], [], []):
                valid_words.append(model.index_to_key[i])

        with open(f'{mod}.pickle', 'wb') as outfile:
            pickle.dump(model, outfile)
        outfile.close()

        with open(f'{mod}_words.pickle', 'wb') as outfile2:
            pickle.dump(valid_words, outfile2)
        outfile2.close()

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, default = 0)
parser.add_argument('--pos', required=True)
parser.add_argument('--neg', required=False, default="")
parser.add_argument("--n_words", required=False, default=10)
parser.add_argument("--alpha", required=False, default=1)
parser.add_argument("--beta", required=False, default=0.5)
parser.add_argument("--serialize", required=True, default=0.5)




def distance_matrix(words, model):
    l2_arr = np.ndarray(shape=(len(words), len(words)))
    cos_arr = np.ndarray(shape=(len(words), len(words)))

    for i in range(len(words)):
        for j in range(len(words)):
            w_i = model[words[i]]
            w_j = model[words[j]]
            l2_dist = np.linalg.norm(w_i - w_j)
            l2_arr[i][j] = round(l2_dist)
            cos_arr[i][j] = round(model.distance(words[i], words[j]), ndigits=3)

    return l2_arr, cos_arr

w1 = ["car", "gasoline", "alcohol", "party"]
w2 = ["tablet", "battery", "house", "california", "solar"]
w3 = ["hat", "fruit", "computer", "pencil", "dog"]

def generate_clue(pos, neg = [], n_words=10, alpha = 1, beta = 0.5, model_num=0):

    model_name = model_names[model_num]

    with open(f"{model_name}.pickle", "rb") as infile:
        model: KeyedVectors = pickle.load(infile)
    infile.close()
    
    with open(f'{model_name}_words.pickle', 'rb') as infile2:
        valid_words = pickle.load(infile2)
    infile2.close()

    pos_vectors = list(map(lambda x: model[x], pos))
    neg_vectors = list(map(lambda x: model[x], neg))

    best = [("", float('inf'))]*n_words

    for word in valid_words:

        if filter_word(word, pos, neg):
            vec = model[word]
            loss = compute_loss(vec, pos_vectors, neg_vectors, alpha, beta)

            if loss < best[-1][1]:
                best[-1] = (word, loss)
            
            best.sort(key=lambda x:x[1])
    
    return best




def compute_loss(vec, pos, neg, alpha, beta):
    running_loss = 0
    
    for v in pos:
        #dist = np.linalg.norm(v - vec)
        dist = 1 - cosine_similarity(v, vec)
        running_loss += alpha*(dist**2)

    for v in neg:
        #dist = np.linalg.norm(v - vec)
        dist = 1 - cosine_similarity(v, vec)
        running_loss -= beta*(dist**2)

    return running_loss

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    return dot/(n1*n2)


if __name__ == '__main__':

    args = parser.parse_args()
    model_num = int(args.model)
    pos = args.pos
    neg = args.neg
    n_words = int(args.n_words)
    alpha = args.alpha
    beta = args.beta    
    p = pos.split()
    n = neg.split()

    if args.serialize:
        serialize_models()

    print(generate_clue(pos=p, neg=n, n_words=n_words, alpha=alpha, beta=beta, model_num=model_num))


