import nltk
from nltk.stem import PorterStemmer
import numpy as np
from tensorflow.keras import models
import json
import tensorflow as tf

def get_labels():
    with open("label_dict.json", "r") as file:
        label_dict = json.load(file)

    return label_dict

def get_model():
    model = tf.keras.models.load_model("model.keras", compile=False)

# If needed, recompile with a new optimizer and loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_vocab_stems():
    with open("vocab.txt", "r") as file:
        vocab_stem_list = [line.strip() for line in file]
    return vocab_stem_list

def preprocess(txt, vocab):
    bags = []

    tokens = nltk.word_tokenize(txt)

    stemmer = PorterStemmer()

    tokens = [stemmer.stem(t.lower()) for t in tokens]

    bag = []
    for t in vocab:
        if t in tokens:
            bag.append(1)
        else:
            bag.append(0)
        
    bags.append(bag)
    bags = np.array(bags)
    print(bags.shape)
    bag_reshape = bags.reshape((-1, 1, bags.shape[1]))
    return bag_reshape

vocab = get_vocab_stems()
model = get_model()
label_dict = get_labels()

user_input = 'How much caffeine is in a double shot compared to a single shot of espresso?'
processed_input = [preprocess(user_input, vocab)]
results = model.predict(processed_input)
print(results)
results_index = np.argmax(results)
tag = label_dict[str(results_index)]

print(tag)


