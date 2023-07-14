# download Glove and load it into torch.nn.Embedding and store the word2index in a dictionary
import torch
import numpy as np
from torchtext.vocab import GloVe
import torch.nn as nn 
import pickle
glove = GloVe(name='42B', dim=300)
word2index = glove.stoi
embedding_matrix = torch.zeros((len(word2index), 300))
embedding = nn.Embedding.from_pretrained(glove.vectors, freeze=False)

pickle.dump([embedding, word2index], open('/w2v/glove.300d.en.txt.pickle', 'wb'))