from pytrie import SortedStringTrie
import requests
import random
import json

class RhymingDictionary:

  def __init__(self, training_data):
    # Assign instance variables
    self.training_data = training_data
    # Read the CMU Pronoucing Dictionary
    # The syllables and pronounciation of the words are extracted based off the layout in dictionary.txt
    self.dictionary = open("dictionary.txt", "r")

  # Count number of syllables in each word
  def count_syllables(self, word):
    numbers = ['0','1','2', '3', '4', '5', '6', '7', '8', '9']
    syllables = 0
    for char in word:
      if char in numbers:
        syllables += 1
    return syllables

  # Create a trie which contains the number of syllables, pronounciation and stresses of words
  def create_syllable_guide(self):
    syllable_count = SortedStringTrie()
    syllable_pronounciation = SortedStringTrie()
    for line in self.dictionary:
      index_space = line.find(" ")
      word = line[:index_space]
      pronounciation = line[index_space + 2 :]
      syllables = self.count_syllables(pronounciation)
      syllable_count.__setitem__(word, syllables)
      syllable_pronounciation.__setitem__(word, pronounciation)
    return syllable_count, syllable_pronounciation

  # Remove extension of words that are not found in the syllable guide, ie 'st, 'er
  def remove_extension(self, word):
    if len(word) > 3:
      if word[-3:] == '\'st' or word[-3:] == '\'er' or word[-3:] == '\'ry':
        word = word[:len(word)-3]
    return word

  # Build the dictionary of words from the data
  # Creates a rhyming dictionary by entering the pronounciation of the last syllable
  def create_rhyming_dictionary(self, syllable_pronounciation):
    data = open(self.training_data, "r")
    rhyming_dictionary = {}
    for line in data:
      words = line.split()
      for word in words:
        real_word = self.remove_extension(word).upper()
        if real_word in syllable_pronounciation:
          pronounciation = syllable_pronounciation[real_word]
          last_occurence_syllable_unstressed = pronounciation.rfind("0")
          last_occurence_syllable_stressed = pronounciation.rfind("1")
          if last_occurence_syllable_stressed > last_occurence_syllable_unstressed:
            sub_pronounciation = pronounciation[:last_occurence_syllable_stressed]
          else:
            sub_pronounciation = pronounciation[:last_occurence_syllable_unstressed]
          index_space = sub_pronounciation.rfind(" ")
          last_syllable = pronounciation[index_space + 1:]
          if last_syllable in rhyming_dictionary:
            current_list = rhyming_dictionary[last_syllable]
            if word not in current_list:
              current_list.append(word)
              rhyming_dictionary[last_syllable] = current_list
          else:
            rhyming_dictionary[last_syllable] = [word]
    return rhyming_dictionary

  # Get rhyme from rhyming dictionary which match the syllable requirements
  def rhyme_pick(self, rhyme_sound, rhyming_dictionary, syllables_required, syllable_count, syllable_pronounciation):
    rhymes = []
    words = rhyming_dictionary[rhyme_sound]
    if syllables_required % 2 == 0:
      for word in words:
        word_alt = self.remove_extension(word)
        if word_alt != word:
          word_alt = word_alt.upper()
          if syllable_count[word_alt] + 1 == syllables_required and syllable_pronounciation[word_alt].find("0") > syllable_pronounciation[word_alt].find("1"):
            rhymes.append(word)
        else:
          word_alt = word_alt.upper()
          if syllable_count[word_alt] == syllables_required and syllable_pronounciation[word_alt].find("0") > syllable_pronounciation[word_alt].find("1"):
            rhymes.append(word)
    else:
      for word in words:
        word_alt = self.remove_extension(word)
        if word_alt != word:
          word_alt = word_alt.upper()
          if syllable_count[word_alt] + 1 == syllables_required and syllable_pronounciation[word_alt].find("0") < syllable_pronounciation[word_alt].find("1"):
            rhymes.append(word)
        else:
          word_alt = word_alt.upper()
          if syllable_count[word_alt] == syllables_required and syllable_pronounciation[word_alt].find("0") < syllable_pronounciation[word_alt].find("1"):
            rhymes.append(word)
    return rhymes

  # If there is no rhyme from the rhyming dictionary, use the RhymeBrain API to find a rhyme
  def rhyme_brain(self, rhyme_word, syllables_required, syllable_pronounciation):
    rhymes = []
    parameters = {"function" : "getRhymes" , "word" : rhyme_word}
    request = requests.get("http://rhymebrain.com/talk", params=parameters)
    json = request.json()
    if (syllables_required % 2 == 0):
      for item in json:
        word = item["word"]
        syllables = item["syllables"]
        if word.upper() in syllable_pronounciation and int(syllables) == syllables_required:
          if syllable_pronounciation[word.upper()].find("1") > syllable_pronounciation[word.upper()].find("0"):
            rhymes.append(word)
    else:
      for item in json:
        word = item["word"]
        syllables = item["syllables"]
        if word.upper() in syllable_pronounciation and int(syllables) == syllables_required:
          if syllable_pronounciation[word.upper()].find("1") < syllable_pronounciation[word.upper()].find("0"):
            rhymes.append(word)
    return rhymes
