import numpy as np
import json
import tensorflow as tf
import nltk
from nltk.stem import PorterStemmer

with open('dataset.json', 'r') as file:
    dataset = json.load(file)

token_set = []
X = []
y = []

labels = []

for intent in dataset['intents']:
    label = intent['tag']
    labels.append(label)
    for q in intent['questions']:
        tokens = nltk.word_tokenize(q)
        token_set.extend(tokens)
        X.append(tokens)
        y.append(label)

print(X[:5])
print(y[:5])

stemmer = PorterStemmer()

token_stems = []
for token in token_set:
    if token is not "?":
        token_stems.append(stemmer.stem(token.lower()))

# print(token_stems)
token_stems = sorted(list(set(token_stems)))
labels = sorted(labels)

# print(token_stems)
# print(labels)

X_bags = []
y_labels = []

for idx, x in enumerate(X):
    bag_of_words = []
    tokens = [stemmer.stem(w.lower()) for w in x]
    print(tokens)
    for t in token_set:
        if t in tokens:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)

    X_bags.append(bag_of_words)
    y_labels.append(labels.index(y[idx]))

# print(X_bags.shape)
# print(y_labels.shape)
