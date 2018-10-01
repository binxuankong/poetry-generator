# For Python 2.7 to work using Python 3 module
from future.standard_library import install_aliases
install_aliases()

import sys
import numpy
from GeneratePoem import *
from GenerateBigram import *
from RNNTheano import RNN
from GRUTheano import GRU
from RhymingDictionary import RhymingDictionary
from argparse import ArgumentParser

training_data = "shakespeare.csv"

# numpy.random.seed(10)

# Set the CL args
parser = ArgumentParser()
usage = "usage: %prog [args] arg1 arg2 arg3"

parser.add_argument("-m", "--mode", help = "select mode from RNN|GRU|RNNBigram|GRUBigram", dest = "mode", choices = ["RNN", "GRU", "RNNBigram", "GRUBigram"])
parser.add_argument("-i", "--iteration", help = "select number of iterations from 20|40|60|80|100", dest = "iteration", choices = ["20", "40", "60", "80", "100"])
parser.add_argument("-t", "--theme", help = "theme of the poem to be generated", dest = "theme")

args = parser.parse_args()

# Must at least provide mode and iteration
if not args.mode or not args.iteration:
  parser.error("Must provide mode and number of iterations")
  sys.exit(0)

number_of_iterations = int(args.iteration)

rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary = generate_rhyming_dictionary(training_data)
dictionary, dictionary_size, word_to_index, start_train, end_train, cantor_dictionary, cantor_dictionary_size, cantor_to_index, start_bigram_train, end_bigram_train, starting_bigrams = load_data(training_data, syllable_count)

# Generate the neural network mode
# RNN
if args.mode == "RNN":
  print("Generating RNN Model...")
  model = RNN(dictionary_size)
  if number_of_iterations == 20:
    model.load_model_parameters('TrainedModels/RNNModel20.npz')
  elif number_of_iterations == 40:
    model.load_model_parameters('TrainedModels/RNNModel40.npz')
  elif number_of_iterations == 60:
    model.load_model_parameters('TrainedModels/RNNModel60.npz')
  elif number_of_iterations == 80:
    model.load_model_parameters('TrainedModels/RNNModel80.npz')
  else:
    model.load_model_parameters('TrainedModels/RNNModel100.npz')
# GRU
elif args.mode == "GRU":
  print("Generating GRU Model...")
  model = GRU(dictionary_size)
  if number_of_iterations == 20:
    model.load_model_parameters('TrainedModels/GRUModel20.npz')
  elif number_of_iterations == 40:
    model.load_model_parameters('TrainedModels/GRUModel40.npz')
  elif number_of_iterations == 60:
    model.load_model_parameters('TrainedModels/GRUModel60.npz')
  elif number_of_iterations == 80:
    model.load_model_parameters('TrainedModels/GRUModel80.npz')
  else:
    model.load_model_parameters('TrainedModels/GRUModel100.npz')
# RNN Bigram
elif args.mode == "RNNBigram":
  print("Generating RNN Bigram Model...")
  model = RNN(cantor_dictionary_size)
  if number_of_iterations == 20:
    model.load_model_parameters('TrainedModels/RNNBigram20.npz')
  elif number_of_iterations == 40:
    model.load_model_parameters('TrainedModels/RNNBigram40.npz')
  elif number_of_iterations == 60:
    model.load_model_parameters('TrainedModels/RNNBigram60.npz')
  elif number_of_iterations == 80:
    model.load_model_parameters('TrainedModels/RNNBigram80.npz')
  else:
    model.load_model_parameters('TrainedModels/RNNBigram100.npz')
# GRU Bigram
else:
  model = GRU(cantor_dictionary_size)
  print("Generating GRU Bigram Model...")
  if number_of_iterations == 20:
    model.load_model_parameters('TrainedModels/GRUBigram20.npz')
  elif number_of_iterations == 40:
    model.load_model_parameters('TrainedModels/GRUBigram40.npz')
  elif number_of_iterations == 60:
    model.load_model_parameters('TrainedModels/GRUBigram60.npz')
  elif number_of_iterations == 80:
    model.load_model_parameters('TrainedModels/GRUBigram80.npz')
  else:
    model.load_model_parameters('GRUBigram100.npz')
print("Done generating library and model")

# If a theme is provided
if args.theme:
  theme = args.theme
  print("Generating poem of theme \"" + theme + "\"...\n")
  try:
    # Normal mode
    if args.mode == "RNN" or args.mode == "GRU":
      poem = generate_poem_theme(model, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, theme)
    # Bigram mode
    else:
      poem = generate_poem_cantor_theme(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_bigrams, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, theme)
    print_poem(poem)
  except(NoRhymeError, KeyError):
    print("Rhyming error occured, no suitable rhyme could be found for the generated poem")
# Generate standard poem
else:
  print("Generating standard poem...\n")
  try:
    # Normal mode
    if args.mode == "RNN" or args.mode == "GRU":
      poem = generate_poem(model, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
    # Bigram mode
    else:
      poem = generate_poem_cantor(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_bigrams, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
    print_poem(poem)
  except(NoRhymeError, KeyError):
    print("Rhyming error occured, no suitable rhyme could be found for the generated poem")
