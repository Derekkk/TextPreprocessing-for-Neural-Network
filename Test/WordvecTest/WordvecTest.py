# -*- encoding: utf-8 -*-

from textPreprocessing.WordvecLoad import WordvecLoad

if __name__ == '__main__':
    wordvec = WordvecLoad("../TokenizerTest/model.vec")

    wordindex = wordvec.word_index
    embeddingmatrix = wordvec.embedding_matrix
    embeddingindex = wordvec.embedding_index

    for word in wordindex.keys():
        index = wordindex[word]
        vec_1 = embeddingindex[word]
        vec_2 = embeddingmatrix[index]
        if (vec_1 == vec_2).all():
            print word
        else:
            print word, index


