import tensorlow as tf

import tkinter as tk
from tkinter import ttk

import json
import random
import numpy as np
import pickle

from tf.keras.models import load_model
from tk.keras.utils import get_file

import nltk
from nltk.stem import WordNetLemmatizer

author_dict = {
		"garcia_marquez" : 'https://gist.githubusercontent.com/ismaproco/6781d297ee65c6a707cd3c901e87ec56/raw/20d3520cd7c53d99215845375b1dca16ac827bd7/gabriel_garcia_marquez_cien_annos_soledad.txt', 
		"frank_baum" : 'http://www.textfiles.com/etext/FICTION/wizrd_oz', 
		"jules_vernes" : 'http://www.textfiles.com/etext/FICTION/under_sea', 
		"berkley" : 'http://www.textfiles.com/etext/NONFICTION/berkeley-three-745.txt'
}

author_model = {"garcia_marquez": "./data/authors/garcia_marquez.h5",  
				"frank_baum": "./data/authors/frank_baum.h5", 
				"jules_vernes": "./data/authors/jules_vernes.h5", 
				"berkley": "./data/authors/berkley.h5" 
}

model = load_model("./data/chatbot_mind.h5")
intents = json.loads(open("./data/intents.json").read())
words = pickle.load(open("./data/words.pkl", "rb"))
classes = pickle.load(open("./data/classes.pkl", "rb"))

def clean_sentence(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [WordNetLemmatizer().lemmatize(word.lower()) for word in sentence_words]
	return sentence_words

def bow(sentence, words, show_details=True):
	sentence_words = clean_sentence(sentence)
	bag = [0] * len(words)
	for s in sentence_words:
		for i, w in enumerate(words):
			if w == s:
				bag[i] = 1
				if show_details:
					print(f"Found in bag: {w}")
	return (np.array(bag))

def predict_class(sentence, model):
	p = bow(sentence, words, show_details=False)
	res = model.predict(np.array([p]))[0]
	ERROR_THRESHOLD = 0.25

	results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []

	for r in results:
		return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

	return return_list

def get_response(ints, intents_json):
	tag = ints[0]['intent']
	list_intents = intents_json['intents']

	for i in list_intents:
		if (i['tag'] == tag):
			result = random.choice(i['responses'])
			break
	return result

def chatbot_response(text):
	ints = predict_class(text, model)
	res = get_response(ints, intents)
	return res

def sample(preds, temperature=1):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

class Application(tk.Tk):
	"""Application root window"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.title("InsPyro v.0.0.1")
		self.geometry("400x500")
		self.resizable(width=False, height=False)
		self.configure(background="black")

		self.image = tk.PhotoImage(file="data/images/icon.png")	
		self.iconphoto(False, self.image)
		
		self.chatlog = tk.Text(self, bd=0, bg="#42433F", height=8, width=50, font="Arial", cursor="trek")
		self.chatlog.config(state=tk.DISABLED)

		self.scrollbar = tk.Scrollbar(self, command=self.chatlog.yview)
		self.chatlog['yscrollcommand'] = self.scrollbar.set

		self.entrybox = tk.Text(self, bd=0, bg="#42433F",width="29", height="5", font="Arial", cursor="pencil")

		self.send_button = tk.Button(self, font=("Arial", 12, "bold"), text="Send", width=10, height=5, bd=0, bg="#960000", activebackground="#FF1418", fg="#ffffff", command=self.send)

		self.generate_combobox = ttk.Combobox(self, width=11, height=2, values=list(author_dict.keys()), textvariable=tk.StringVar())
		self.generate_combobox.set("Choose...")

		self.generate_button = tk.Button(self, font=("Times", 12, "bold"), text="Inspire", width=11, height=3, bd=0, bg="#960000", activebackground="#FF1418", fg="#ffffff", command= self.chatbot_generate)

		self.scrollbar.place(x=378,y=6, height=386)
		self.chatlog.place(x=7,y=6, height=386, width=370)
		self.entrybox.place(x=128, y=401, height=90, width=265)
		self.send_button.place(x=6, y=401, height=40)
		self.generate_button.place(x=6, y=445, height=20)
		self.generate_combobox.place(x=6, y=468, height=20)


	def send(self):
		msg = self.entrybox.get("1.0", 'end-1c').strip()
		self.entrybox.delete("0.0", tk.END)

		if msg != '':
			self.chatlog.config(state=tk.NORMAL)
			self.chatlog.insert(tk.END, "You: " + msg + "\n\n")
			self.chatlog.config(foreground="#FF3600", font=("Arial", 12))

			res = chatbot_response(msg)
			self.chatlog.insert(tk.END, "InsPyro: " + res + "\n\n")

			self.chatlog.config(state=tk.DISABLED)
			self.chatlog.yview(tk.END)

	#TODO Generate text for the user
	def chatbot_generate(self, model=None, lenght=100):
		author = self.generate_combobox.get()
		max_len = 60
		file_path = get_file(f"{author}.txt", author_dict[author])
		file = open(file_path).read().lower()
		chars = sorted(list(set(file)))

		if author != '':
			if author in list(author_dict.keys()):
				model = load_model(author_model[author])
				self.chatlog.config(state=tk.NORMAL)
				self.chatlog.insert(tk.END, "InsPyro:  Inspiring..." + "\n\n")
				generated = ''
				start_index = random.randint(0, len(file) - max_len - 1)
				sentence = file[start_index: start_index + max_len]
				generated += sentence

				for i in range(length):
					sampled = np.zeros((1, max_len, len(chars)))
					
					for t, char in enumerate(sentence):
						sampled[0, t, char_index[char]] = 1.
					preds = model.predict(sampled, verbose=0)[0]
					next_index = sample(preds, temperature=0.8)
					next_char = chars[next_index]

					generated += next_char
					sentence = sentence[1:] + next_char

				self.chatlog.insert(tk.END, "InsPyro: " + generated + "\n\n")

				self.chatlog.config(state=tk.DISABLED)
				self.chatlog.yview(tk.END)
			else:
				self.chatlog.insert(tk.END, "InsPyro: Theres no author with that name in the database.")
				self.chatlog.config(state=tk.DISABLED)
				self.chatlog.yview(tk.END)
		else:
			self.chatlog.insert(tk.END, "InsPyro: No option was selected.")
			self.chatlog.config(state=tk.DISABLED)
			self.chatlog.yview(tk.END)

if __name__== "__main__":
	app = Application()
	app.mainloop()
