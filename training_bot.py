import numpy as np
import random
from tensorflow import keras
import json
import pprint
import pickle

import nltk
from nltk.stem import WordNetLemmatizer

#Load data and convert words into its lemma form
print("LOADING THE DATA...")

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open("data/intents.json").read()

intents = json.loads(data_file)

for intent in intents['intents']:
	for pattern in intent['patterns']:
		w = nltk.word_tokenize(pattern)
		words.extend(w)
		documents.append((w, intent['tag']))

		[classes.append(intent['tag']) if intent['tag'] not in classes else None]

words = [WordNetLemmatizer().lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

words_file = open('./data/words.pkl', 'wb')
classes_file = open('./data/classes.pkl', 'wb')

pickle.dump(words, words_file)
pickle.dump(classes, classes_file)

#Create training data to train the model
print("CREATING TRAINING SET...")

training = []
output_empty = [0] * len(classes)

for doc in documents:
	bag = []
	pattern_words = doc[0]

	pattern_words = [WordNetLemmatizer().lemmatize(word.lower()) for word in pattern_words]

	for w in words:
		bag.append(1) if w in pattern_words else bag.append(0)

	output_row = list(output_empty)
	output_row[classes.index(doc[1])] = 1

	training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

X_train = list(training[:,0])
y_train = list(training[:,1])

X_train = np.array(X_train)
y_train = np.array(y_train)

#Building the model
print("BUILDING AND TRAINING THE MODEL...")

model = keras.models.Sequential([
	keras.layers.Dense(128, activation='relu', input_shape=(len(X_train[0]),)),
	keras.layers.Dropout(0.5),
	keras.layers.Dense(64, activation='relu'),
	keras.layers.Dropout(0.5),
	keras.layers.Dense(len(y_train[0]), activation='softmax')
])

model.compile(optimizer=keras.optimizers.SGD(lr=0.01, nesterov=True, momentum=0.9, decay=1e-6), 
	loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, y_train, epochs=200, batch_size=5)

model.save('./data/chatbot_mind.h5', hist)

print("Model has been saved succesfully at data folder.")