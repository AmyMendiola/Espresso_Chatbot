import nltk
import random
from nltk.stem import PorterStemmer
import numpy as np
from tensorflow.keras import models
import json
import tensorflow as tf
from nltk.corpus import wordnet
import user_db

#get label idx dictionary (created in 'train' file)
#ARGS: None, RETURNS: label dictionary {"idx": "label"}
def get_labels():
    with open("label_dict.json", "r") as file:
        label_dict = json.load(file)

    return label_dict

#get dataset from dataset.json
#ARGS: None, RETURNS: json loaded dataset
def get_dataset():
    with open('dataset.json', 'r') as file:
        dataset = json.load(file)
    
    return dataset

#load in the trained model (trained in 'train.ipynb')
#ARGS: None, RETURNS: trained model
def get_model():
    model = tf.keras.models.load_model("model.keras", compile=False)

    #recompile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#load in the stemmed vocab from the data set (created in 'train.ipynb')
#ARGS: None RETURNS: stemmed vocab array
def get_vocab_stems():
    with open("vocab.txt", "r") as file:
        vocab_stem_list = [line.strip() for line in file]
    return vocab_stem_list

#preprocess the incoming user input for a model prediction
#ARGS: txt: user input, vocab: db stemmed vocab, RETURNS: preprocessed/reshaped user txt
def preprocess(txt, vocab):
    bags = []

    #tokenize
    tokens = nltk.word_tokenize(txt)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t.lower()) for t in tokens]

    #create bag of words
    bag = []
    for t in vocab:
        if t in tokens:
            bag.append(1)
        else:
            bag.append(0)    
    bags.append(bag)

    #reshape for model
    bags = np.array(bags)
    bag_reshape = bags.reshape((-1, 1, bags.shape[1]))
    return bag_reshape

#aquires synonym set for a word
#ARGS: word, RETURNS: synonym array
def get_syn(w):
    for s in wordnet.synsets(w):
        syn_arr = []
        for l in s.lemmas():
            syn_arr.append(l.name())
    return syn_arr

#generates a random greeting string
#ARGS: None, RETURN: greeting(string)
def greet_user():
    greeting = ', I am Espresso Information Chatbot designed to give general information and help you learn about espresso.'

    #aquire random greeting phrase
    hello_syn = get_syn('Hello')
    rand = random.randint(0, len(hello_syn)-1)
    rand_hello = hello_syn[rand]

    greeting = rand_hello + greeting
    return greeting

#generates a random farewell string
#ARGS: none, RETURN: farewell(string)
def farewell():
    farewell = '! ' + user_tag + ', if you have any more espresso related questions feel free to ask another time.'

    #aquire random goodbye phrase
    bye_syn = get_syn('bye')
    rand = random.randint(0, len(bye_syn)-1)
    rand_bye = bye_syn[rand]

    farewell = rand_bye + farewell
    return farewell

#maps precited tag to dataset response
#ARGS: tag: model predicted tag, dataset: dataset, RETURNS: db response (string)
def get_response(tag, dataset):
    for t in dataset['intents']:
        if t['tag'] == tag:
            response = t['responses'][0]
            
    return response


dataset = get_dataset()
vocab = get_vocab_stems()
model = get_model()
label_dict = get_labels()
user_db.init()


#*** START BOT ***
user_tag = 'USER' #stating user tag
bot_fulltag = 'BOT: '


#*** GREET ***
greeting = greet_user()
print ('\n\n\n\n'+ greeting)


#*** PROCESS NAME ***
inquire_name = bot_fulltag + 'Before we start, can I ask what your name is?'
print (inquire_name) #prompt name
name = input(user_tag + ": ")
user, db, isReturning = user_db.find(name) #check if user un user model
user_tag = user['name'].upper()

if (isReturning == False):
    #prompt previous knowledge
    inquire_coffee = bot_fulltag + 'May I also ask you to rate on a scale of 1(little) to 5(lots) your knowledge on espresso and coffee? '
    print(inquire_coffee)
    user_input = input(user_tag + ": ")
    user_db.update(name, "prior knowledge score", user_input) #save previous knowledge

    greeting_tag = 'Nice to meet you, '
else: 
    greeting_tag = 'Welcome Back, '


#*** BOT INTRO ***
print(bot_fulltag + greeting_tag + user_tag + '!')
print(bot_fulltag + 'If you ever want to stop the conversation, reply with \'!\'.')
print(bot_fulltag + 'Let me introduce you to some of the topics you can learn about. You can ask about ', end='')
for i, topic in enumerate(label_dict.values()): #loop for dataset topics
    if i != len(label_dict.values()) - 1:
        print(topic, end=', ')
    else:
        print('and ' + topic + '.')


#*** CONVERSATION LOOP ***
keep_talking = True
while (keep_talking): #conversation loop
    user_input = input(user_tag + ": ")
    if (user_input == '!'): #check for end conversation
        keep_talking = False

    else: #process and predict on user question
        processed_input = [preprocess(user_input, vocab)]
        results = model.predict(processed_input,verbose = 0)
        results_index = np.argmax(results)
        tag = label_dict[str(results_index)]

        #user model update
        user_db.update(name, "interested topics", tag)
        user_db.update(name, "queries", user_input)

        #print response
        print(bot_fulltag + get_response(tag, dataset))


#*** FINAL Q ***
#check if favorite drink hasnt been answered, if not ask
if user_db.get(name, "favorite drink") == "":
    print(bot_fulltag + user_tag + ' before you go, Do you have a favorite coffee or espresso drink? If so, what is it?')
    fav_drink = input(user_tag + ": ")
    if (fav_drink.lower() != 'no' and fav_drink.lower() != 'n'):
        user_db.update(name, "favorite drink", fav_drink) #update user model: yes
        print(bot_fulltag + fav_drink + ' is a very cool choice.')
    else: user_db.update(name, "favorite drink", "N/A") #update user model: no


#*** BYE BYE ***
print(bot_fulltag + farewell())
print('\nThank You!\n')


#*** DISPLAY USER DATA ***
#display user model items for the current user
print('Here are the user logged items: ')
user, db, _ = user_db.find(name)
print(user)

