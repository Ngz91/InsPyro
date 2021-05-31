import tensorflow as tf
import numpy as np
import random
from tf.keras.models import load_model
from tf.keras.utils import get_file

author_dict = {"tolstoy" : 'http://www.textfiles.com/etext/FICTION/war_peace_text', 
		"garcia_marquez" : 'https://gist.githubusercontent.com/ismaproco/6781d297ee65c6a707cd3c901e87ec56/raw/20d3520cd7c53d99215845375b1dca16ac827bd7/gabriel_garcia_marquez_cien_annos_soledad.txt', 
		"sir_richard_burton" : 'http://www.textfiles.com/etext/FICTION/burton-arabian-363.txt', 
		"frank_baum" : 'http://www.textfiles.com/etext/FICTION/wizrd_oz', 
		"jules_vernes" : 'http://www.textfiles.com/etext/FICTION/under_sea', 
		"berkley" : 'http://www.textfiles.com/etext/NONFICTION/berkeley-three-745.txt', 
		"hume" : 'http://www.textfiles.com/etext/NONFICTION/hume-enquiry-65.txt', 
		"hobbes" : 'http://www.textfiles.com/etext/NONFICTION/leviathan', 
		"spinoza" : 'http://www.textfiles.com/etext/NONFICTION/spinoza-theologico-743.txt'
}

author_model = {"tolstoy": "./authors/tolstoy.h5", 
				"garcia_marquez": "./authors/garcia_marquez.h5", 
				"richard_burton": "./authors/richard_burton.h5", 
				"frank_baum": "./authors/frank_baum.h5", 
				"jules_vernes": "./authors/jules_vernes.h5", 
				"berkley": "./authors/berkley.h5", 
				"hume": "./authors/hume.h5", 
				"hobbes": "./authors/hobbes.h5", 
				"spinoza": "./authors/spinoza.h5"}

def file_path(name, path):
	f_path = get_file(name, path)
	return f_path

def sample(preds, temperature=1):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

file_p = file_path('berkley.txt', author_dict["berkley"])
file = open(file_p).read().lower()

chars = sorted(list(set(file)))
max_len = 80

temperatures = [0.2, 0.5, 0.7, 1.0]

def text_generate(modelo, length=60):
	model = load_model(modelo)
	generated = ''
	start_index = random.randint(0, len(file) - max_len - 1)
	sentence = file[start_index: start_index + max_len]
	generated += sentence

	temperatures = [0.2, 0.5, 0.6, 0.8, 1.0]

	for temperature in temperatures:
		for i in range(length):
			sampled = np.zeros((1, max_len, len(chars)))
			
			for t, char in enumerate(sentence):
				sampled[0, t, char_index[char]] = 1.
			preds = model.predict(sampled, verbose=0)[0]
			next_index = sample(preds, temperature)
			next_char = chars[next_index]

			generated += next_char
			sentence = sentence[1:] + next_char

	return generated			

text_generate("./authors/berkley.h5", lenght=200)