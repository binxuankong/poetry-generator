# For Python 2.7 to work using Python 3 module
from future.standard_library import install_aliases
install_aliases()

import csv
import itertools
import collections
import math
import nltk
import numpy
import operator
import sys
import random
import requests
import json
import pronouncing
from urllib.request import urlopen
from GeneratePoem import NoRhymeError
from RhymingDictionary import RhymingDictionary

number_lines = 14
line_length = [6, 7, 8, 9, 10]
line_start_token = "line_start"
line_end_token = "line_end"
unknown_token = "unknown_token"

# List of words to never end a line with
banned_end_words = ['a', 'ah', 'an', 'and', 'at', 'are', 'as', 'but', 'by', 'has', 'i', 'is', 'in', 'my', 'no', 'not', 'o', 'of', 'or', 's', 'than', 'the', 'their', 'then', 'to', 'was', 'with', 'your', line_start_token, line_end_token, unknown_token]

# The Big Huge Thesaurus API Key
bht_api = 'acf445020c3778040589b030b88031a1'

# Remove extension of words that are not found in the syllable guide, ie 'st, 'er
def remove_extension(word):
  if len(word) > 3:
    if word[-3:] == '\'st' or word[-3:] == '\'er' or word[-3:] == '\'ry':
      word = word[:len(word)-3]
  return word

# Return a list of bigrams given a list
def find_bigrams(input_list):
  return list(zip(input_list, input_list[1:]))

# Cantor pairing function
def cantor_pair(x, y):
  z = (x + y + 1) * (x + y)
  z = (z / 2) + y
  return int(z)

# Apply the cantor pairing function to each tuple in a list
def cantor_pair_list(input_list):
  output_list = []
  for (x, y) in input_list:
    output_list.append(cantor_pair(x, y))
  return output_list

# Invert the cantor pairing function
def invert_cantor(z):
  w = math.sqrt(8 * z + 1) - 1
  s = math.floor(w / 2)
  t = (s * s + s) / 2
  y = int(z - t)
  x = int(s - y)
  return (x, y)

# Apply the inverse of cantor pairing function to a list
def invert_cantor_list(input_list):
  output_list = []
  for z in input_list:
    output_list.append(invert_cantor(z))
  return output_list

# Generate the rhyming dictionary
def generate_rhyming_dictionary(training_data):
  print("Generating rhyming dictionary...")
  rhyming_model = RhymingDictionary(training_data)
  syllable_count, syllable_pronounciation = rhyming_model.create_syllable_guide()
  rhyming_dictionary = rhyming_model.create_rhyming_dictionary(syllable_pronounciation)
  return rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary

# Read and process the training data
def load_data(training_data, syllable_count):
  # Read the data
  print("Reading %s..." %training_data)
  with open(training_data, mode='r') as o:
    reader = csv.reader(o, skipinitialspace = True)
    # Add start and end token at each line
    lines = itertools.chain(*[x for x in reader])
    lines = ["%s %s %s" %(line_start_token, x.lower(), line_end_token) for x in lines]
  print("Parsed %d lines." %len(lines))
  # Tokenize each line into words
  tokenized_lines =  [nltk.word_tokenize(x) for x in lines]
  # Get dictionary
  words = collections.Counter(itertools.chain(*tokenized_lines))
  dictionary = []
  dictionary.append(line_start_token)
  dictionary.append(line_end_token)
  dictionary.append(unknown_token)
  # Only add the word if it is found in the rhyming dictionary
  for word in words:
    word_alt = remove_extension(word)
    if word_alt.upper() in syllable_count:
      dictionary.append(word)
  dictionary_size = len(dictionary)
  print("Using vocabulary size %d." %dictionary_size)
  # Replace all words not in the dictionary with the unknown token
  for i, line in enumerate(tokenized_lines):
    tokenized_lines[i] = [w if w in dictionary else unknown_token for w in line]
  word_to_index = dict([(w,i) for i,w in enumerate(dictionary)])
  # Create training data
  start_train = numpy.asarray([[word_to_index[w] for w in line[:-1]] for line in tokenized_lines])
  end_train = numpy.asarray([[word_to_index[w] for w in line[1:]] for line in tokenized_lines])
  # Create templates for training data for bigrams
  start_cantor_train = []
  for i, line in enumerate(start_train):
    start_cantor_train.append(cantor_pair_list(find_bigrams(line)))
  end_cantor_train = []
  for i, line in enumerate(end_train):
    end_cantor_train.append(cantor_pair_list(find_bigrams(line)))
  # Change the templates to a flat list
  flat_start_cantor = [item for sublist in start_cantor_train for item in sublist]
  flat_end_cantor = [item for sublist in end_cantor_train for item in sublist]
  # Combine the start and end templates to one dictionary
  start_cantor_set = set(flat_start_cantor)
  end_cantor_set = set(flat_end_cantor)
  cantor_dictionary = list(start_cantor_set.union(end_cantor_set))
  cantor_to_index = dict([(c, i) for i, c in enumerate(cantor_dictionary)])
  cantor_dictionary_size = len(cantor_dictionary)
  print("Using bigram size %d." %cantor_dictionary_size)
  # Actual training data for cantors
  start_cantor_train1 = numpy.asarray([[cantor_to_index[c] for c in line] for line in start_cantor_train])
  end_cantor_train1 = numpy.asarray([[cantor_to_index[c] for c in line] for line in end_cantor_train])
  # Get the starting bigrams of each line
  starting_cantors = []
  for cantor in start_cantor_train1:
    starting_cantor = cantor[0]
    starting_bigram = invert_cantor(starting_cantor)
    if word_to_index[unknown_token] not in starting_bigram:
      starting_cantors.append(starting_cantor)
  return dictionary, dictionary_size, word_to_index, start_train, end_train, cantor_dictionary, cantor_dictionary_size, cantor_to_index, start_cantor_train1, end_cantor_train1, starting_cantors

# Get the syllables of the given word
def get_syllables(word, dictionary, syllable_count):
  word_alt = remove_extension(dictionary[word])
  if word_alt.upper() in syllable_count:
    if word_alt == dictionary[word]:
      return syllable_count[word_alt.upper()]
    else:
      return syllable_count[word_alt.upper()] + 1
  else:
    return 0

# Get the next word using forward propagation
# Try to get a word from the theme if possible
def get_next_cantor(model, line, syllables_remaining, dictionary, word_to_index, cantor_dictionary, cantor_to_index, syllable_count, keywords_in_dictionary):
  sampled_bigram = (word_to_index[unknown_token], word_to_index[unknown_token])
  sampled_cantor = cantor_to_index[cantor_pair(*sampled_bigram)]
  sampled_word = sampled_bigram[1]
  sampled_word_syllables = 0
  next_word_probs = model.forward_propagation(line)
  # A theme is given, try to replace with given theme words
  if keywords_in_dictionary != None:
    while sampled_word_syllables == 0 or sampled_word_syllables > syllables_remaining:
      samples = numpy.random.multinomial(1, next_word_probs[-1])
      sampled_cantor = numpy.argmax(samples)
      sampled_bigram = invert_cantor(cantor_dictionary[sampled_cantor])
      sampled_word = sampled_bigram[1]
      if dictionary[sampled_word] == line_end_token or dictionary[sampled_word] == unknown_token:
        sampled_word = word_to_index[random.choice(keywords_in_dictionary)]
      sampled_word_syllables = get_syllables(sampled_word, dictionary, syllable_count)
  # No theme is given, repeat while bigram contains the line end or unknown token
  else:
    while sampled_word_syllables == 0 or sampled_word_syllables > syllables_remaining or word_to_index[line_end_token] in sampled_bigram or word_to_index[unknown_token] in sampled_bigram:
      samples = numpy.random.multinomial(1, next_word_probs[-1])
      sampled_cantor = numpy.argmax(samples)
      sampled_bigram = invert_cantor(cantor_dictionary[sampled_cantor])
      sampled_word = sampled_bigram[1]
      sampled_word_syllables = get_syllables(sampled_word, dictionary, syllable_count)
  return sampled_cantor, sampled_word

# Get a rhyming word given a word
def get_rhyme(rhyme_word, syllables_required, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary):
  rhyme_word_alt = remove_extension(rhyme_word)
  if rhyme_word_alt.upper() in syllable_pronounciation:
    pronounciation = syllable_pronounciation[rhyme_word_alt.upper()]
    last_occurence_syllable_unstressed = pronounciation.rfind("0")
    last_occurence_syllable_stressed = pronounciation.rfind("1")
    if last_occurence_syllable_stressed > last_occurence_syllable_unstressed:
      sub_pronounciation = pronounciation[:last_occurence_syllable_stressed]
    else:
      sub_pronounciation = pronounciation[:last_occurence_syllable_unstressed]
    space_index = sub_pronounciation.rfind(" ")
    last_syllable = pronounciation[space_index + 1:]
    rhymes = rhyming_model.rhyme_pick(last_syllable, rhyming_dictionary, syllables_required, syllable_count, syllable_pronounciation)
    for rhyme in rhymes:
      if rhyme in banned_end_words or rhyme == rhyme_word:
        rhymes.remove(rhyme)
    if len(rhymes) >= 1:
      rhyme = random.choice(rhymes)
    else:
      # No rhymes can be found from the rhyming dictionary, use RhymeBrain
      rhymes = rhyming_model.rhyme_brain(rhyme_word, syllables_required, syllable_pronounciation)
      if len(rhymes) >= 1:
        rhyme = random.choice(rhymes)
      # No rhymes can be found from RhymeBrain, use pronouncing
      else:
        rhymes = pronouncing.rhymes(rhyme_word)
        if len(rhymes) >= 1:
          rhyme = random.choice(rhymes)
        else:
          raise NoRhymeError("Cannot find a rhyming word.")
  else:
    # No rhymes can be found from the rhyming dictionary, use RhymeBrain
    rhymes = rhyming_model.rhyme_brain(rhyme_word, syllables_required, syllable_pronounciation)
    if len(rhymes) >= 1:
      rhyme = random.choice(rhymes)
    else:
      # No rhymes can be found from RhymeBrain, use pronouncing
      rhymes = pronouncing.rhymes(rhyme_word)
      if len(rhymes) >= 1:
        rhyme = random.choice(rhymes)
      else:
        raise NoRhymeError("Cannot find a rhyming word.")
  return rhyme.lower()

# Generate line cantor
def generate_line_cantor(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, syllable_count, keywords_in_dictionary=None):
  # Start the line with a random starting cantor
  new_line = [random.choice(starting_cantors)]
  syllables_remaining = 10
  syllable_count = 0
  # Get the actual line from the bigram
  starting_bigram = invert_cantor(new_line[0])
  starting_word = starting_bigram[1]
  actual_line = [starting_word]
  syllables_remaining = syllables_remaining - get_syllables(starting_word, dictionary, syllable_count)
  while syllables_remaining > 0:
    next_cantor = get_next_cantor(model, new_line, syllables_remaining, dictionary, word_to_index, cantor_dictionary, cantor_to_index, syllable_count, keywords_in_dictionary)
    new_line.append(next_cantor[0])
    actual_line.append(next_cantor[1])
    syllables_remaining = syllables_remaining - next_cantor[2]
  # Convert vectors back to words using dictionary
  line = [dictionary[x] for x in actual_line]
  # Randomly replaces nouns in the sentence if a theme is given
  if keywords_in_dictionary != None:
    line_temp = " ".join(line)
    line_pos = nltk.pos_tag(nltk.word_tokenize(line_temp))
    counter = 0
    for pos in line_pos:
      if "NN" in pos[1]:
        random_number = numpy.random.randint(4)
        if random_number == 1:
          random_theme_word = random.choice(keywords_in_dictionary)
          if get_syllables(word_to_index[pos[0]], dictionary, syllable_count) == get_syllables(word_to_index[random_theme_word], dictionary, syllable_count):
            line[counter] = random_theme_word
      counter = counter + 1
  return line

# Ensure that the lines end with a rhyme (without theme)
def couplet(x, y, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary):
  line_1 = lines[x]
  line_2 = lines[y]
  end_word_1 = line_1[-1]
  end_word_2 = line_2[-1]
  # Replace the second end word with a word that rhymes with the first end word
  if end_word_1 not in banned_end_words:
    end_word_1_alt = remove_extension(end_word_1)
    if end_word_1_alt.upper() in syllable_pronounciation:
      syllables_required = get_syllables(word_to_index[end_word_2], dictionary, syllable_count)
      line_2[-1] = get_rhyme(end_word_1, syllables_required, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  # Replace the first end word with a word that rhymes with the second end word
  elif end_word_2 not in banned_end_words:
    end_word_2_alt = remove_extension(end_word_2)
    if end_word_2_alt.upper() in syllable_pronounciation:
      syllables_required = get_syllables(word_to_index[end_word_1], dictionary, syllable_count)
      line_1[-1] = get_rhyme(end_word_2, syllables_required, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  # Both the first and second end word are in the banned end word list
  # Select two random rhyming words to use as the end words
  else:
    while end_word_1 not in banned_end_words:
      end_word_1 = random.choice(dictionary)
    end_word_1_alt = remove_extension(end_word_1)
    if end_word_1_alt.upper() in syllable_pronounciation:
      syllables_required = get_syllables(word_to_index[end_word_2], dictionary, syllable_count)
      line_2[-1] = get_rhyme(end_word_1, syllables_required, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  return [line_1, line_2]

# Ensure that the lines end with a rhyme
# Try to use a word from theme if possible
def couplet_theme(x, y, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, keywords):
  line_1 = lines[x]
  line_2 = lines[y]
  end_word_1 = line_1[-1]
  end_word_2 = line_2[-1]
  end_word_1_alt = remove_extension(end_word_1)
  end_word_2_alt = remove_extension(end_word_2)
  # If either end word is in the baneed end words list, replace with a theme word
  if end_word_1 in banned_end_words or end_word_2 in banned_end_words:
    if end_word_1 in banned_end_words:
      end_word_1 = random.choice(keywords)
      end_word_1_alt = remove_extension(end_word_1)
      if end_word_1_alt.upper() in syllable_pronounciation:
        syllables_required = get_syllables(word_to_index[end_word_2], dictionary, syllable_count)
        line_2[-1] = get_rhyme(end_word_1, syllables_required, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
    else:
      end_word_2 = random.choice(keywords)
      end_word_2_alt = remove_extension(end_word_2)
      if end_word_2_alt.upper() in syllable_pronounciation:
        syllables_required = get_syllables(word_to_index[end_word_1], dictionary, syllable_count)
        line_1[-1] = get_rhyme(end_word_2, syllables_required, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  else:
    if end_word_1_alt.upper() in syllable_pronounciation:
      syllables_required = get_syllables(word_to_index[end_word_2], dictionary, syllable_count)
      line_2[-1] = get_rhyme(end_word_1, syllables_required, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  return [line_1, line_2]

# Generate lines with the rhyming scheme ABABCDCDEFEFGG (without theme)
def generate_lines_cantor(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary):
  lines = []
  for i in range(number_lines):
    this_line = generate_line_cantor(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, syllable_count)
    lines.append(this_line)
  c1 = couplet(0, 2, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  c2 = couplet(1, 3, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  c3 = couplet(4, 6, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  c4 = couplet(5, 7, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  c5 = couplet(8, 10, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  c6 = couplet(9, 11, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  c7 = couplet(12, 13, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  return [c1[0], c2[0], c1[1], c2[1], c3[0], c4[0], c3[1], c4[1],
          c5[0], c6[0], c5[1], c6[1], c7[0], c7[1]]

# Generate lines with the rhyming scheme ABABCDCDEFEFGG (with theme)
def generate_lines_cantor_theme(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, keywords, keywords_in_dictionary):
  lines = []
  for i in range(number_lines):
    this_line = generate_line_cantor(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, syllable_count, keywords_in_dictionary)
    lines.append(this_line)
  c1 = couplet_theme(0, 2, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, keywords)
  c2 = couplet_theme(1, 3, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, keywords)
  c3 = couplet_theme(4, 6, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, keywords)
  c4 = couplet_theme(5, 7, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, keywords)
  c5 = couplet_theme(8, 10, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, keywords)
  c6 = couplet_theme(9, 11, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, keywords)
  c7 = couplet_theme(12, 13, lines, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, keywords)
  return [c1[0], c2[0], c1[1], c2[1], c3[0], c4[0], c3[1], c4[1],
          c5[0], c6[0], c5[1], c6[1], c7[0], c7[1]]

# Generate a poem with no theme
def generate_poem_cantor(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary):
  poem = generate_lines_cantor(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  return poem

# Generate a poem with a given theme
# Find synonyms of the given theme using the Big Huge Thesaurus
def generate_poem_cantor_theme(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, theme):
  keywords = []
  keywords_in_dictionary = []
  url = "http://api.datamuse.com/words?ml=" + theme
  response = urlopen(url)
  data = json.load(response)
  words = [theme]
  for related_word in data:
    if 'n' in related_word['tags']:
      words.append(related_word['word'])
  keywords_count = 0
  for word in words:
    if word.upper() in syllable_count:
      keywords.append(word)
    if word in dictionary:
      keywords_in_dictionary.append(word)
      keywords_count += 1
    if keywords_count > 8:
      break
  if len(keywords_in_dictionary) == 0:
    print("Unfortunately no poem could be generated using the theme \"" + theme + "\". Generating a standard poem instead...")
    poem = generate_lines_cantor(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  else:
    poem = generate_lines_cantor_theme(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, keywords, keywords_in_dictionary)
  return poem, keywords

# Generate a poem with a given image
# Find synonyms of the three keywords obtained from the image using the Big Huge Thesaurus
def generate_poem_cantor_image(model, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, image_words):
  keywords = []
  keywords_in_dictionary = []
  for image_word in image_words:
    url = "http://api.datamuse.com/words?ml=" + image_word
    response = urlopen(url)
    data = json.load(response)
    words = [image_word]
    for related_word in data:
      if 'n' in related_word['tags']:
        words.append(related_word['word'])
    keywords_count = 0
    for word in words:
      if word.upper() in syllable_count:
        keywords.append(word)
      if word in dictionary:
        keywords_in_dictionary.append(word)
        keywords_count += 1
      if keywords_count > 6:
        break
  if len(keywords_in_dictionary) == 0:
    print("Unfortunately no poem could be generated based on the image. Generating a standard poem instead...")
    poem = generate_lines_cantor(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
  else:
    poem = generate_lines_cantor_theme(model, dictionary, word_to_index, cantor_dictionary, cantor_to_index, starting_cantors, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, keywords, keywords_in_dictionary)
  return poem, keywords

# Print out the poem
def print_poem(poem):
  for i in range(number_lines):
    print(" ".join(poem[i]))
