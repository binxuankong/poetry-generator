# For Python 2.7 to work using Python 3 module
from future.standard_library import install_aliases
install_aliases()

import numpy
import matplotlib.pyplot as plt
from GeneratePoem import *
from GenerateBigram import *
from RNNTheano import RNN
from GRUTheano import GRU
from RhymingDictionary import RhymingDictionary
from argparse import ArgumentParser

training_data = "shakespeare.csv"

# Set the CL args
parser = ArgumentParser()
usage = "usage: %prog [args] arg1 arg2 arg3"

parser.add_argument("-m", "--mode", help = "select mode from RNN|GRU|RNNBigram|GRUBigram", dest = "mode", choices = ["RNN", "GRU", "RNNBigram", "GRUBigram"])
parser.add_argument("-i", "--iteration", type = int, dest = "iteration")
parser.add_argument("-f", "--filename", dest = "filename")

args = parser.parse_args()

if not args.mode or not args.iteration or not args.filename:
  parser.error("Must provide exactly 3 arguments")
  sys.exit(0)

number_of_iterations = int(args.iteration)
save_to_file = args.filename

# Load the dictionary
rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary = generate_rhyming_dictionary(training_data)
dictionary, dictionary_size, word_to_index, start_train, end_train, cantor_dictionary, cantor_dictionary_size, cantor_to_index, start_bigram_train, end_bigram_train, starting_bigrams = load_data(training_data, syllable_count)

# For graph plotting
loss_array = []
num_examples_array = []

# Train the model
# RNN
if args.mode == "RNN":
  print("Generating RNN Model...")
  model = RNN(dictionary_size)
  print("Training RNN Model with %d iterations..." %number_of_iterations)
  losses = model.train_with_sgd(start_train, end_train, nepoch = number_of_iterations)
  for (num_examples, loss) in losses:
    num_examples_array.append(num_examples)
    loss_array.append(loss)
  plt.plot(num_examples_array, loss_array)
  plt.title('Losses of Trained RNN Model %d iterations' %number_of_iterations)
  plt.xlabel('number of examples seen')
  plt.ylabel('losses')
  plt.grid(True)
  plt.show()
  model.save_model_parameters(save_to_file)
  print("Done training model")
# GRU
elif args.mode == "GRU":
  print("Generating GRU Model...")
  model = GRU(dictionary_size)
  print("Training GRU Model with %d iterations..." %number_of_iterations)
  losses = model.train_with_sgd(start_train, end_train, nepoch = number_of_iterations)
  for (num_examples, loss) in losses:
    num_examples_array.append(num_examples)
    loss_array.append(loss)
  plt.plot(num_examples_array, loss_array)
  plt.title('Losses of Trained GRU Model %d iterations' %number_of_iterations)
  plt.xlabel('number of examples seen')
  plt.ylabel('losses')
  plt.grid(True)
  plt.show()
  model.save_model_parameters(save_to_file)
  print("Done training model")
# RNN Bigram
elif args.mode == "RNNBigram":
  print("Generating RNN Bigram Model...")
  model = RNN(cantor_dictionary_size)
  print("Training RNN Bigram Model with %d iterations..." %number_of_iterations)
  losses = model.train_with_sgd(start_bigram_train, end_bigram_train, nepoch = number_of_iterations)
  for (num_examples, loss) in losses:
    num_examples_array.append(num_examples)
    loss_array.append(loss)
  plt.plot(num_examples_array, loss_array)
  plt.title('Losses of Trained RNN Model %d iterations' %number_of_iterations)
  plt.xlabel('number of examples seen')
  plt.ylabel('losses')
  plt.grid(True)
  plt.show()
  model.save_model_parameters(save_to_file)
  print("Done training model")
# GRU Bigram
else:
  print("Generating GRU Bigram Model...")
  model = GRU(cantor_dictionary_size)
  print("Training GRU Bigram Model with %d iterations..." %number_of_iterations)
  losses = model.train_with_sgd(start_bigram_train, end_bigram_train, nepoch = number_of_iterations)
  for (num_examples, loss) in losses:
    num_examples_array.append(num_examples)
    loss_array.append(loss)
  plt.plot(num_examples_array, loss_array)
  plt.title('Losses of Trained GRU Model %d iterations' %number_of_iterations)
  plt.xlabel('number of examples seen')
  plt.ylabel('losses')
  plt.grid(True)
  plt.show()
  model.save_model_parameters(save_to_file)
  print("Done training model")
