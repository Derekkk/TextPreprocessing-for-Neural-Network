"""
Author: Derek Hu
Date: 2018/ 04/ 26
Function: Tokenizer for embedding, to solve OOV problem in prediction
"""
# -*- encoding: utf-8 -*-
import jieba

class Tokenzier(object):
    """
    num_words: the max number of words want to be kept
    word_index: dict, with key to be the token and value to be the index
    """
    def __init__(self, num_words=None, word_index={}):
        self.num_words = num_words
        self.word_index = word_index

    def text_to_sequence(self, texts):
        """Transforms each text in texts in a sequence of integers."""
        text_sequeces = []
        for text in texts:
            ts = []
            for i in range(0, len(text)):
                value = self.word_index.get(text[i].encode('utf-8'))
                if value is not None:
                    ts.append(value)
                else:
                    ts.append(0)
            text_sequeces.append(ts)
        return text_sequeces

    def pad_sequence(self, sequences):
        """pad sequence to the given length, with 0"""
        result = []
        for sequence in sequences:
            length = min(self.num_words, len(sequence))
            seq = sequence[0:length]
            pad_length = self.num_words - len(seq)
            for i in range(0, pad_length):
                seq.insert(0, 0)
            result.append(seq)
        return result
