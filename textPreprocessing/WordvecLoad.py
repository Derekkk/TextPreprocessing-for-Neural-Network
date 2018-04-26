"""
Author: Derek Hu
Date: 2018/ 04/ 26
Function: load pre-trained word2vec model
"""

# -*- encoding: utf-8 -*-
import numpy as np


class WordvecLoad(object):
    """
    embedding_index: { word: vec }
    word_index: { word: index}
    embedding_matrix: { index: vec }
    """
    def __init__(self, path=None, Embedding_dim = 0):
        self.path = path
        self.EMBEDDING_DIM = Embedding_dim
        #self.word_index = {}
        #self.embedding_index = {}
        #self.embedding_matrix = {}

        """Read pre-trained word vector"""
        f = open(self.path, 'r')
        embedding_index = {}  # embedding metrix, key is word and value is vector
        for line in f.readlines()[1:]:
            value = line.split()
            word = value[0]
            vec = np.asarray(value[1:])
            embedding_index[word] = vec
        f.close()
        self.embedding_index = embedding_index

        """compute word_index: { word: index}"""
        wordindex = {}
        embeddingmatrix = np.zeros((len(self.embedding_index) + 1, self.EMBEDDING_DIM))
        index = 1
        for item in self.embedding_index.items():
            assert len(item) == 2
            wordindex[item[0]] = index
            embeddingmatrix[index] = item[1]
            index += 1
        self.word_index = wordindex
        self.embedding_matrix = embeddingmatrix

