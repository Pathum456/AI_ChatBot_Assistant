import json
import random
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset

# nltk.download('punkt')

class Preprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def tokenize(self, sentence):
        # Directly tokenize the input sentence without sentence splitting
        return nltk.tokenize.word_tokenize(sentence)

    def stem(self, word):
        return self.stemmer.stem(word.lower())

    def bag_of_words(self, tokenized_sentence, all_words):
        tokenized_sentence = [self.stem(w) for w in tokenized_sentence]
        bag = np.zeros(len(all_words), dtype=np.float32)
        for idx, w in enumerate(all_words):
            if w in tokenized_sentence:
                bag[idx] = 1.0
        return bag


class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples