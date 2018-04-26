# TextPreprocessing-for-Neural-Network
### This is a simple implementation of text preprocessing, including text2sequence, padding, and embedding matrix construction.

Only for python2. If you want to use in python3, there are several encoding issues to be fixed.

### How to use it
Here is an example:

"""
from textPreprocessing.Tokenizer import Tokenzier
from textPreprocessing.WordvecLoad import WordvecLoad

wordvec = WordvecLoad("model.vec", 256)
tokenizer = Tokenzier(num_words=MAX_SENTENCE_LENGTH, word_index=wordvec.word_index)
data = tokenizer.text_to_sequence(data)
data = tokenizer.pad_sequence(data)

print "Split data into train and test set"
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
print len(X_train), len(X_test)
print len(y_train), len(y_test)

print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)
"""


The reason why I create this is that the existed preprocessing function is not very convenient to use. Also, just want to make things easier. If there are any problem or bugs or suggestions, plz feel free to contact me.
