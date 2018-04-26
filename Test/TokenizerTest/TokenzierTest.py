# -*- encoding: utf-8 -*-
import jieba

from textPreprocessing.Tokenizer import Tokenzier
from textPreprocessing.WordvecLoad import WordvecLoad

if __name__ == '__main__':
    texts = ["今天天气咋样", "你好啊", "我想办理信用卡", "请问一下如果我要办理信用卡需要什么软件"]
    texts_1 = []
    for text in texts:
        text = list(jieba.cut(text))
        texts_1.append(text)
    print(texts_1)

    # word_index = {}
    # word_index["今天"] = 1
    # word_index["天气"] = 2
    # word_index["你好"] = 3
    # word_index["信用卡"] = 4
    # word_index["请问"] = 5
    wordvec = WordvecLoad("wordvec.txt", Embedding_dim=256)
    tokenzer = Tokenzier(num_words=5, word_index=wordvec.word_index)
    f = open('text.txt', 'w')
    for item in tokenzer.word_index.iteritems():
        f.write(str(item[0])+'\t'+str(item[1])+'\n')

    text = tokenzer.text_to_sequence(texts_1)
    print(text)
    text = tokenzer.pad_sequence(text)
    print(text)
    print(len(wordvec.word_index), len(wordvec.embedding_index), len(wordvec.embedding_matrix))
    for item in wordvec.word_index.items():
        print item[0], item[1]
    print len(wordvec.embedding_matrix)
