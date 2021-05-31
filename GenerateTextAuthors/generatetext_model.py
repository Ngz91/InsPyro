import numpy as np
from tensorflow import keras
import random

author_dict = {"tolstoy" : 'http://www.textfiles.com/etext/FICTION/war_peace_text', 
		"garcia_marquez" : 'https://gist.githubusercontent.com/ismaproco/6781d297ee65c6a707cd3c901e87ec56/raw/20d3520cd7c53d99215845375b1dca16ac827bd7/gabriel_garcia_marquez_cien_annos_soledad.txt', 
		"richard_burton" : 'http://www.textfiles.com/etext/FICTION/burton-arabian-363.txt', 
		"frank_baum" : 'http://www.textfiles.com/etext/FICTION/wizrd_oz', 
		"jules_vernes" : 'http://www.textfiles.com/etext/FICTION/under_sea', 
		"berkley" : 'http://www.textfiles.com/etext/NONFICTION/berkeley-three-745.txt', 
		"hume" : 'http://www.textfiles.com/etext/NONFICTION/hume-enquiry-65.txt', 
		"hobbes" : 'http://www.textfiles.com/etext/NONFICTION/leviathan', 
		"spinoza" : 'http://www.textfiles.com/etext/NONFICTION/spinoza-theologico-743.txt'
}

#Preparing the data
def file_path(name, path):
	f_path = keras.utils.get_file(name, path)
	return f_path

print("Preparing the data...")

file_p = file_path('berkley.txt', author_dict["berkley"])
file = open(file_p).read().lower()
print(f"Lenght: {len(file)}")

max_len = 60
step = 3
sentences = []
next_chars = []

for i in range(0, len(file) - max_len, step):
	sentences.append(file[i: i + max_len])
	next_chars.append(file[i + max_len])

print(f"Number of sequences: {len(sentences)}")

chars = sorted(list(set(file)))

print(f"Unique Characters: {len(chars)}")

char_index = dict((char, chars.index(char)) for char in chars)
print("Start of Vectorization...")

#Shape of Sequences, maxlen, unique_characters
x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		x[i, t, char_index[char]] = 1
	y[i, char_index[next_chars[i]]] = 1

#Building and training the model
print("BUILDING AND TRAINING THE MODEL")

model = keras.models.Sequential([
	keras.layers.LSTM(128, input_shape=(max_len, len(chars))),
	keras.layers.Dense(len(chars), activation="softmax")
])

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.01), loss="categorical_crossentropy", callbacks=keras.callbacks.EarlyStopping(patience=16, restore_best_weights=True, monitor='loss')

model.fit(x, y, batch_size=148, epochs=80)

model.save("./authors/berkley.h5")

print("DONE")