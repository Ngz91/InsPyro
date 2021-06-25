InsPyro
=======

Description
===========

InsPyro is an Assistant bot build in Python using Recurrent Neural Networks and NLP. The idea behind InsPyro is simple, generate a text of what the author might have writen, a text that could be of use for writers. Right now InsPyro is incomplete, and there are some features that do not work as expected, since i'm still learning NLP. In the future I might migrate the project from using NLTK to using spaCy since I have found that the later offers me more options.

Features
========

* An interactive UI for ease of use.
* Write to the bot and it will write back, try to ask it for it's name, it's age, or the meaning of life.
* Text Generation based on famous works from various authors.

Authors
=======

Nevio Gomez, 2021

Requirements
============

* Python 3
* Tkinter
* NLTK
* pickle
* Keras 2.3.0 (model build with Tensorflow)
* Jupyter Notebook

Usage
=====

* To talk to the bot, write on the gray textbox and click enter, the bot will interact with you.

* To generate a text, choose an autor from the combobox (right now there are only models available for Gabriel Garcia Marquez, Berkley, Frank Baum and Jules Vernes) and click Inspire, the bot will generate a text based on a work from the selected author (right now due to limitations in hardware, this feature has not yet being tested, but i believe that there are no errors in the code).
