import numpy as np
import json
import tensorflow as tf
import nltk
from nltk.stem import PorterStemmer
import keras
from tensorflow.keras import layers

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

X_bags = np.array(X_bags)
y_labels = np.array(y_labels)
# print(X_bags.shape)
# print(y_labels.shape)

model = keras.Sequential([
    layers.InputLayer(input_shape=(X_bags.shape[1], 1)),  # Bag-of-words shape with 1 feature
    layers.LSTM(32),  # LSTM with 32 units
    layers.Dense(16, activation='relu'),  # A dense layer
    layers.Dense(15, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_bags_reshaped = X_bags.reshape((X_bags.shape[0], X_bags.shape[1], 1))

model.fit(X_bags_reshaped, y_labels, epochs=10, validation_data=(X_bags_reshaped, y_labels))
