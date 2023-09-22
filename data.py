from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('../data/ner_datasetreference.csv', encoding='latin1')
enc = OrdinalEncoder()
tag_col = enc.fit_transform(data['Tag'].to_numpy().reshape(-1, 1))
tag_col = pd.Series(tag_col.reshape(-1,))
data['Tag'] = tag_col

RANGES = {'TRAIN': (0, 700198),
          'VALID': (700199, 875421),
          'TEST': (875422, 1048574)}



def adjust_tokens_by_1(data, decrease):
    if decrease:
        diff = -1
    else:
        diff = 1

    for sentence_id in range(len(data)):
        for word_id in range(len(data[sentence_id])):
            data[sentence_id][word_id] += diff


def one_hot_set(data, depth):
    for sentence_id in range(len(data)):
        data[sentence_id] = tf.one_hot(data[sentence_id], depth)


def convert_set_to_ragged_tensor(data):
    list_of_tensors = []
    for sample in data:
        list_of_tensors.append(tf.convert_to_tensor(sample, dtype=tf.float32))
    return tf.ragged.stack(list_of_tensors)


def preprocess_sentences(list_of_lists, tag_list):
    list_of_strings = []
    indexes_to_pop_from_y = []
    for list_of_words_idx in range(len(list_of_lists)):
        try:
            list_of_strings.append(' '.join(list_of_lists[list_of_words_idx]))
        except TypeError:
            print('Missing word - sentence will be omitted')
            tag_list.pop(list_of_words_idx)
    return list_of_strings, tag_list


def get_set(set_type):
    set_range = RANGES[set_type]
    ds = data.loc[set_range[0]:set_range[1], :]
    ds['Sentence #'] = ds['Sentence #'].ffill()
    sentences_and_tags = ds.groupby('Sentence #')[['Word', 'Tag']].agg(list)
    return preprocess_sentences(sentences_and_tags['Word'].tolist(), sentences_and_tags['Tag'].tolist())


X_train, y_train = get_set('TRAIN')
X_valid, y_valid = get_set('VALID')
X_test, y_test = get_set('TEST')
print('Checkpoint #1')
n_words = 500
categories = 17
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=n_words, oov_token='<OOV>', filters='')
tokenizer.fit_on_texts(X_train)
print('Checkpoint #2')
X_train = tokenizer.texts_to_sequences(X_train)
X_valid = tokenizer.texts_to_sequences(X_valid)
X_test = tokenizer.texts_to_sequences(X_test)
print('Checkpoint #3')
adjust_tokens_by_1(X_train, True)
adjust_tokens_by_1(X_valid, True)
adjust_tokens_by_1(X_test, True)
print('Checkpoint #4')
one_hot_set(X_train, n_words)
one_hot_set(X_valid, n_words)
one_hot_set(X_test, n_words)
print('Checkpoint #5')
X_train = convert_set_to_ragged_tensor(X_train)
X_valid = convert_set_to_ragged_tensor(X_valid)
X_test = convert_set_to_ragged_tensor(X_test)
print('Checkpoint #6')
y_train = convert_set_to_ragged_tensor(y_train)
y_valid = convert_set_to_ragged_tensor(y_valid)
y_test = convert_set_to_ragged_tensor(y_test)
print('Checkpoint #7')