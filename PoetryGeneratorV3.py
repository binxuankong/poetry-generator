import os
import csv
import itertools
import nltk
import urllib
import random
from GeneratePoem import *
from GenerateBigram import *
from RNNTheano import RNN
from GRUTheano import GRU
from RhymingDictionary import RhymingDictionary
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

training_data = "shakespeare.csv"

# Clarifai App for image classification
clarifai_app = ClarifaiApp(api_key='bd31f96760af4dbc9b3112d9e58b1ce3')
clarifai_model = clarifai_app.models.get('general-v1.3')

# Required for showing examples of Shakespeare's Sonnets
example_data = open('shakespeare.txt')
lines = example_data.readlines()
# Tokenized lines for BLEU score
with open('shakespeare.csv', mode='r') as o:
  reader = csv.reader(o, skipinitialspace = True)
  csv_lines = itertools.chain(*[x for x in reader])
  csv_lines = [x.lower() for x in csv_lines]
tokenized_lines =  [nltk.word_tokenize(x) for x in csv_lines]
reference = [x for x in tokenized_lines]

# Index for log message
global logger_index
logger_index = 0.0
# Check if generate using bigram or not
bigram_flag = False
# Base width and height of the image
base_width = 440
base_height = 440

# Load rhyming dictionary
rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary = generate_rhyming_dictionary(training_data)
dictionary, dictionary_size, word_to_index, start_train, end_train, cantor_dictionary, cantor_dictionary_size, cantor_to_index, start_bigram_train, end_bigram_train, starting_bigrams = load_data(training_data, syllable_count)
model = RNN(dictionary_size)

# Load the premade RNN model based on the number of iterations
def load_rnn():
  iterations = number_iterations.get()
  if iterations == 20:
    filename = "RNNModel20.npz"
  elif iterations == 40:
    filename = "RNNModel40.npz"
  elif iterations == 60:
    filename = "RNNModel60.npz"
  elif iterations == 80:
    filename = "RNNModel80.npz"
  else:
    filename = "RNNModel100.npz"
  model = RNN(dictionary_size)
  log_message = model.load_model_parameters("TrainedModels/" + filename)
  generate_sonnet1_button['state'] = 'normal'
  generate_sonnet2_button['state'] = 'normal'
  generate_sonnet1_image_button['state'] = 'normal'
  generate_sonnet2_image_button['state'] = 'normal'
  bigram_flag = False
  write_to_log(log_message + " iterations=%d" %iterations, logger_index)
  current_model.set("Current Model: RNN %d iterations" %iterations)

# Load the premade GRU model based on the number of iterations
def load_gru():
  iterations = number_iterations.get()
  if iterations == 20:
    filename = "GRUModel20.npz"
  elif iterations == 40:
    filename = "GRUModel40.npz"
  elif iterations == 60:
    filename = "GRUModel60.npz"
  elif iterations == 80:
    filename = "GRUModel80.npz"
  else:
    filename = "GRUModel100.npz"
  model = GRU(dictionary_size)
  log_message = model.load_model_parameters("TrainedModels/" + filename)
  generate_sonnet1_button['state'] = 'normal'
  generate_sonnet2_button['state'] = 'normal'
  generate_sonnet1_image_button['state'] = 'normal'
  generate_sonnet2_image_button['state'] = 'normal'
  bigram_flag = False
  write_to_log(log_message + " iterations=%d" %iterations, logger_index)
  current_model.set("Current Model: GRU %d iterations" %iterations)

# Load the premade RNN Bigram model based on the number of iterations
def load_rnn_bigram():
  iterations = number_iterations.get()
  if iterations == 20:
    filename = "RNNBigram20.npz"
  elif iterations == 40:
    filename = "RNNBigram40.npz"
  elif iterations == 60:
    filename = "RNNBigram60.npz"
  elif iterations == 80:
    filename = "RNNBigram80.npz"
  else:
    filename = "RNNBigram100.npz"
  model = RNN(cantor_dictionary_size)
  log_message = model.load_model_parameters("TrainedModels/" + filename)
  generate_sonnet1_button['state'] = 'normal'
  generate_sonnet2_button['state'] = 'normal'
  generate_sonnet1_image_button['state'] = 'normal'
  generate_sonnet2_image_button['state'] = 'normal'
  bigram_flag = True
  write_to_log(log_message + " iterations=%d" %iterations, logger_index)
  current_model.set("Current Model: RNN (Bigram) %d iterations" %iterations)

# Load the premade GRU Bigram model based on the number of iterations
def load_gru_bigram():
  iterations = number_iterations.get()
  if iterations == 20:
    filename = "GRUBigram20.npz"
  elif iterations == 40:
    filename = "GRUBigram40.npz"
  elif iterations == 60:
    filename = "GRUBigram60.npz"
  elif iterations == 80:
    filename = "GRUBigram80.npz"
  else:
    filename = "GRUBigram100.npz"
  model = GRU(cantor_dictionary_size)
  log_message = model.load_model_parameters("TrainedModels/" + filename)
  generate_sonnet1_button['state'] = 'normal'
  generate_sonnet2_button['state'] = 'normal'
  generate_sonnet1_image_button['state'] = 'normal'
  generate_sonnet2_image_button['state'] = 'normal'
  bigram_flag = True
  write_to_log(log_message + " iterations=%d" %iterations, logger_index)
  current_model.set("Current Model: GRU (Bigram) %d iterations" %iterations)

# Load a custom RNN model based on the file chosen by the user
def load_custom_rnn():
  root.update()
  filename = filedialog.askopenfilename(initialdir = os.getcwd(), title ='Select file', filetypes=[('Numpy Files', '*.npz')])
  if filename != "":
    try:
      model = RNN(dictionary_size)
      log_message = model.load_model_parameters(filename)
      generate_sonnet1_button['state'] = 'normal'
      generate_sonnet2_button['state'] = 'normal'
      bigram_flag = False
      write_to_log(log_message, logger_index)
      current_model.set("Current Model: Custom RNN")
    except(IOError):
      write_to_log("No such file or directory: %s" %filename, logger_index)
      return

# Load a custom GRU model based on the file chosen by the user
def load_custom_gru():
  root.update()
  filename = filedialog.askopenfilename(initialdir = os.getcwd(), title = 'Select file', filetypes=[('Numpy Files', '*.npz')])
  if filename != "":
    try:
      model = GRU(dictionary_size)
      log_message = model.load_model_parameters(filename)
      generate_sonnet1_button['state'] = 'normal'
      generate_sonnet2_button['state'] = 'normal'
      bigram_flag = False
      write_to_log(log_message, logger_index)
      current_model.set("Current Model: Custom GRU")
    except(IOError):
      write_to_log("No such file or directory: %s" %filename, logger_index)
      return

# Add word by word to display to allow highlight of theme word
def append_to_display(arg, word, theme_words = None):
  if theme_words is None:
    colour_tag = "color-black"
  elif word.lower() in theme_words:
    colour_tag = "color-red"
  else:
    colour_tag = "color-black"
  if arg == 1:
    if theme_words is not None and word.lower() in theme_words:
      poem1_display.tag_configure(colour_tag, foreground = "red")
    else:
      poem1_display.tag_configure(colour_tag, foreground = "black")
    poem1_display['state'] = 'normal'
    poem1_display.insert(END, word + " ", colour_tag)
    poem1_display['state'] = 'disabled'
  else:
    if theme_words is not None and word.lower() in theme_words:
      poem2_display.tag_configure(colour_tag, foreground = "red")
    else:
      poem2_display.tag_configure(colour_tag, foreground = "black")
    poem2_display['state'] = 'normal'
    poem2_display.insert(END, word + " ", colour_tag)
    poem2_display['state'] = 'disabled'

# Update the image of the poem
def update_image(arg, image):
  image_width = image.size[0]
  image_height = image.size[1]
  # If the width of the image is greater than the height
  if image_width > image_height:
    width_percent = base_width / float(image_width)
    height_size = int((float(image_height) * float(width_percent)))
    replacement_image = ImageTk.PhotoImage(image.resize((base_width, height_size), Image.ANTIALIAS))
  # If the height of the image is greater than the width
  else:
    height_percent = base_height / float(image_height)
    width_size = int((float(image_width) * float(height_percent)))
    replacement_image = ImageTk.PhotoImage(image.resize((width_size, base_height), Image.ANTIALIAS))
  if arg == 1:
    poem1_image.configure(image = replacement_image)
    poem1_image.image = replacement_image
  else:
    poem2_image.configure(image = replacement_image)
    poem2_image.image = replacement_image

# Generate a sonnet. If a theme is given, generate a sonnet with theme
# arg is the sonnet number generated
def generate_sonnet(arg):
  try:
    this_theme = ""
    keywords = None
    if arg == 1:
      this_theme = theme1.get()
    else:
      this_theme = theme2.get()
    # Generate poem without theme
    if this_theme == "":
      # Generate poem using bigrams
      if bigram_flag:
        poem = generate_poem_cantor(model, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
      # Generate poem normally
      else:
        poem = generate_poem(model, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary)
    # Generate poem with theme
    else:
      try:
        # Generate poem using bigrams
        if bigram_flag:
          poem, keywords = generate_poem_cantor_theme(model, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, this_theme)
        # Generate poem normally
        else:
          poem, keywords = generate_poem_theme(model, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, this_theme)
      except(urllib.error.HTTPError):
        clear_sonnet(arg)
        write_to_log("Cannot generate a sonnet of theme \"" + this_theme + "\"", logger_index)
    clear_sonnet(arg)
    for sentences in poem:
      i = 0
      for words in sentences:
        if i == 0:
          append_to_display(arg, words.capitalize(), keywords)
        else:
          append_to_display(arg, str(words), keywords)
        i = i + 1
      if arg == 1:
        poem1_display['state'] = 'normal'
        poem1_display.insert(END, "\n")
        poem1_display['state'] = 'disabled'
      else:
        poem2_display['state'] = 'normal'
        poem2_display.insert(END, "\n")
        poem2_display['state'] = 'disabled'
    if this_theme == "":
      write_to_log("Generated a sonnet with no theme", logger_index)
    else:
      write_to_log("Generated sonnet of theme \"" + this_theme + "\"", logger_index)
    # Update the image back to the normal one
    poem_image = Image.open('calligraphy.jpg')
    update_image(arg, poem_image)
  except(NoRhymeError, KeyError) as error:
    clear_sonnet(arg)
    write_to_log("Rhyming error occured, no suitable rhyme could be found for the generated poem", logger_index)
    # Update the image back to the normal one
    poem_image = Image.open('calligraphy.jpg')
    update_image(arg, poem_image)

# Generate a sonnet based on the image given
def generate_sonnet_image(arg):
  root.update()
  filename = filedialog.askopenfilename(initialdir = os.getcwd(), title = 'Select file', filetypes=[('Image files', '*.jpeg *.jpg *.png *.bmp *.gif')])
  if filename != "":
    try:
      image = ClImage(file_obj = open(filename, 'rb'))
      predict = clarifai_model.predict([image])
      image_words = predict['outputs'][0]['data']['concepts']
      no_of_keywords = 0
      counter = 0
      image_keywords = []
      _digits = re.compile('\d')
      def contains_digits(d):
        return bool(_digits.search(d))
      while no_of_keywords < 3:
        this_keyword = image_words[counter]['name']
        if this_keyword != "no person" and not contains_digits(this_keyword):
          image_keywords.append(this_keyword)
          no_of_keywords = no_of_keywords + 1
        counter = counter + 1
      write_to_log("Key word obtained from image: %s, %s, %s" %(image_keywords[0], image_keywords[1], image_keywords[2]), logger_index)
    except(IOError):
      write_to_log("No such file or directory: %s" %filename, logger_index)
      return
    # Now generate a poem using the keyword obtained
    try:
      try:
        # Generate poem using bigrams
        if bigram_flag:
          poem, keywords = generate_poem_cantor_image(model, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, image_keywords)
        # Generate poem normally
        else:
          poem, keywords = generate_poem_image(model, dictionary, word_to_index, rhyming_model, syllable_count, syllable_pronounciation, rhyming_dictionary, image_keywords)
      except(urllib.error.HTTPError):
        clear_sonnet(arg)
        write_to_log("Cannot generate a sonnet based on image \"" + filename + "\"", logger_index)
      clear_sonnet(arg)
      for sentences in poem:
        i = 0
        for words in sentences:
          if i == 0:
            append_to_display(arg, words.capitalize(), keywords)
          else:
            append_to_display(arg, str(words), keywords)
          i = i + 1
        if arg == 1:
          poem1_display['state'] = 'normal'
          poem1_display.insert(END, "\n")
          poem1_display['state'] = 'disabled'
        else:
          poem2_display['state'] = 'normal'
          poem2_display.insert(END, "\n")
          poem2_display['state'] = 'disabled'
      write_to_log("Generated sonnet based on image \"" + filename + "\"", logger_index)
      # Update the poem image
      poem_image = Image.open(filename)
      update_image(arg, poem_image)
    except(NoRhymeError, KeyError) as error:
      clear_sonnet(arg)
      write_to_log("Rhyming error occured, no suitable rhyme could be found for the generated poem", logger_index)

# Get a random Shakespeare's sonnet
def get_example_poem(arg):
  try:
    sonnet_number = random.randint(0, 155)
    starting_line = sonnet_number * 16
    example_poem = ""
    for i in range(starting_line, starting_line + 15):
      example_poem = example_poem + str(lines[i]).lstrip()
    clear_sonnet(arg)
    if arg == 1:
      poem1_display['state'] = 'normal'
      poem1_display.insert(1.0, example_poem)
      poem1_display['state'] = 'disabled'
    else:
      poem2_display['state'] = 'normal'
      poem2_display.insert(1.0, example_poem)
      poem2_display['state'] = 'disabled'
    write_to_log("Retrieved Shakespeare's sonnet #%d" %sonnet_number, logger_index)
    # Update the image back to the normal one
    poem_image = Image.open('calligraphy.jpg')
    update_image(arg, poem_image)
  except(IndexError):
    example_poem = ""
    for i in range(0, 15):
      example_poem = example_poem + str(lines[i]).lstrip()
    clear_sonnet(arg)
    if arg == 1:
      poem1_display['state'] = 'normal'
      poem1_display.insert(1.0, example_poem)
      poem1_display['state'] = 'disabled'
    else:
      poem2_display['state'] = 'normal'
      poem2_display.insert(1.0, example_poem)
      poem2_display['state'] = 'disabled'
    write_to_log("Retrieved Shakespeare's sonnet #1", logger_index)
    # Update the image back to the normal one
    poem_image = Image.open('calligraphy.jpg')
    update_image(arg, poem_image)

# Get the BLEU score of the given sonnet
def bleu_score(arg):
  if arg == 1:
    sampled_poem = poem1_display.get('1.0', 'end')
  else:
    sampled_poem = poem2_display.get('1.0', 'end')
  sampled_poem = sampled_poem.rstrip()
  sampled_poem = sampled_poem.replace(',', '')
  sampled_poem = sampled_poem.split('\n')
  sampled_poem = [x.lower() for x in sampled_poem]
  sampled_token = [nltk.word_tokenize(x) for x in sampled_poem]
  # Smoothing function
  sf = SmoothingFunction().method4
  score = []
  for x in sampled_token:
    score.append(sentence_bleu(reference, x, weights = (1, 0, 0, 0), smoothing_function = sf))
  average_score = sum(score) / len(score)
  write_to_log("BLEU (individual 1-gram) score of Sonnet 1: %.2f" %average_score, logger_index)

# Get the n-gram BLEU score of the given sonnet
def bleu_ngram_score(arg):
  if arg == 1:
    sampled_poem = poem1_display.get('1.0', 'end')
  else:
    sampled_poem = poem2_display.get('1.0', 'end')
  sampled_poem = sampled_poem.rstrip()
  sampled_poem = sampled_poem.replace(',', '')
  sampled_poem = sampled_poem.split('\n')
  sampled_poem = [x.lower() for x in sampled_poem]
  sampled_token = [nltk.word_tokenize(x) for x in sampled_poem]
  # Smoothing function
  sf = SmoothingFunction().method4
  score = []
  for x in sampled_token:
    score.append(sentence_bleu(reference, x, weights = (0.25, 0.25, 0.25, 0.25), smoothing_function = sf))
  average_score = sum(score) / len(score)
  write_to_log("BLEU (cumulative 4-gram) score of Sonnet 1: %.2f" %average_score, logger_index)

# Copy the given sonnet to clipboard
def copy_to_clipboard(arg):
  root.clipboard_clear()
  if arg == 1:
    root.clipboard_append(poem1_display.get('1.0', 'end'))
  else:
    root.clipboard_append(poem2_display.get('1.0', 'end'))
  root.update()
  write_to_log("Copied sonnet to clipboard", logger_index)

# Clear the given sonnet space
def clear_sonnet(arg):
  if arg == 1:
    poem1_display['state'] = 'normal'
    poem1_display.delete(1.0, 'end')
    poem1_display['state'] = 'disabled'
  else:
    poem2_display['state'] = 'normal'
    poem2_display.delete(1.0, 'end')
    poem2_display['state'] = 'disabled'

# Clear log
def clear_log():
  log['state'] = 'normal'
  log.delete(1.0, 'end')
  log['state'] = 'disabled'

# Exit the program
def exit_program(event = None):
  result = messagebox.askyesno(message = "Are you sure you want exit?", icon = 'question', title = 'Exit')
  if result == True:
    root.destroy()
  else:
    result = ""

# Write log
def write_to_log(message, logger_index):
  log['state'] = 'normal'
  logger_index = logger_index + 1.0
  log.insert(logger_index, message + "\n")
  log['state'] = 'disabled'

# Functions for shortcut key binding
# Undo
def shortcut_undo(event):
  root.focus_get().event_generate('<<Undo>>')
# Redo
def shortcut_redo(event):
  root.focus_get().event_generate('<<Redo>>')
# Cut
def shortcut_cut(event):
  root.focus_get().event_generate('<<Cut>>')
# Copy
def shortcut_copy(event):
  root.focus_get().event_generate('<<Copy>>')
# Paste
def shortcut_paste(event):
  root.focus_get().event_generate('<<Paste>>')
# Delete
def shortcut_delete(event):
  root.focus_get().event_generate('<<Delete>>')
# Select All
def shortcut_selectall(event):
  root.focus_get().event_generate('<<SelectAll>>')

def show_about_window():
  top = Toplevel()
  top.title("About Shakespeare's Sonnet Generator")
  title = Message(top, text = "Shakespeare's Sonnet Generator", font = 'helvetica 14 bold')
  about_message = "Shakespeare's Sonnet Generator is an application that generates Shakespeare's styled sonnets using Recurrent Neural Network. \n Created by Bin Xuan Kong \n Supervised by Dr. Tingting Mu \n University of Manchester, BSc(Hons) in Computer Science and Mathematics Third Year Project"
  about = Message(top, text = about_message, justify = CENTER)
  about.pack()
  button = Button(top, text = "Dismiss", command = top.destroy)
  button.pack()

def showHelpDialog():
  return

# ROOT
root = Tk()
root.title("Shakespeare's Sonnet Generator")

# Main frame
main_frame = ttk.Frame(root, padding = (3, 3, 12, 12))
main_frame.grid(column = 0, row = 0, sticky = 'nwes')
main_frame.columnconfigure(0, weight = 1)
main_frame.rowconfigure(0, weight = 1)

# Menu bar
root.option_add('*tearOff', FALSE)
menu_bar = Menu(root)
root['menu'] = menu_bar
# Create menu bars
menu_file = Menu(menu_bar)
menu_edit = Menu(menu_bar)
menu_about = Menu(menu_bar, name = 'linux')
menu_window = Menu(menu_bar, name = 'window')
menu_help = Menu(menu_bar, name = 'help')
menu_iterations = Menu(menu_bar)
# Cascade menu bars to menu
menu_bar.add_cascade(menu = menu_file, label = 'File')
menu_bar.add_cascade(menu = menu_edit, label = 'Edit')
menu_bar.add_cascade(menu = menu_about, label = 'About')
menu_bar.add_cascade(menu = menu_window, label = 'Window')
menu_bar.add_cascade(menu = menu_help, label = 'Help')
# Menu bars commands
# File
menu_file.add_command(label = 'Load RNN Model', command = lambda: load_rnn)
menu_file.add_command(label = 'Load GRU Model', command = lambda: load_gru)
menu_file.add_command(label = 'Load RNN Bigram Model', command = lambda: load_rnn_bigram)
menu_file.add_command(label = 'Load GRU Bigram Model', command = lambda: load_gru_bigram)
menu_file.add_separator()
number_iterations = IntVar()
number_iterations.set(60)
menu_file.add_cascade(menu = menu_iterations, label = 'Number of Iterations')
menu_iterations.add_radiobutton(label = '20 Iterations', variable = number_iterations, value = 20)
menu_iterations.add_radiobutton(label = '40 Iterations', variable = number_iterations, value = 40)
menu_iterations.add_radiobutton(label = '60 Iterations', variable = number_iterations, value = 60)
menu_iterations.add_radiobutton(label = '80 Iterations', variable = number_iterations, value = 80)
menu_iterations.add_radiobutton(label = '100 Iterations', variable = number_iterations, value = 100)
menu_file.add_separator()
menu_file.add_command(label = 'Open RNN Model File', command = load_custom_rnn)
menu_file.add_command(label = 'Open GRU Model File', command = load_custom_gru)
menu_file.add_separator()
menu_file.add_command(label = 'Generate Sonnet 1', command = lambda: generate_sonnet(1))
menu_file.add_command(label = 'Generate Sonnet 2', command = lambda: generate_sonnet(2))
menu_file.add_separator()
menu_file.add_command(label = 'Open Image File Generate Sonnet 1', command = lambda: generate_sonnet_image(1))
menu_file.add_command(label = 'Open Image File Generate Sonnet 2', command = lambda: generate_sonnet_image(2))
menu_file.add_separator()
menu_file.add_command(label = 'Get Example Sonnet 1', command = lambda: get_example_poem(1))
menu_file.add_command(label = 'Get Example Sonnet 2', command = lambda: get_example_poem(2))
menu_file.add_separator()
menu_file.add_command(label = 'Close', command = exit_program, accelerator = 'Ctrl+W')
# Edit
menu_edit.add_command(label = 'Undo', command = shortcut_undo, accelerator = "Ctrl+Z")
menu_edit.add_command(label = 'Redo', command = shortcut_redo, accelerator = "Ctrl+Shift+Z")
menu_edit.add_separator()
menu_edit.add_command(label = 'Cut', command = shortcut_cut, accelerator = "Ctrl+X")
menu_edit.add_command(label = 'Copy', command = shortcut_copy, accelerator = "Ctrl+C")
menu_edit.add_command(label = 'Paste', command = shortcut_paste, accelerator = "Ctrl+V")
menu_edit.add_command(label = 'Delete', command = shortcut_delete)
menu_edit.add_separator()
menu_edit.add_command(label = 'Select All', command = shortcut_selectall, accelerator = "Ctrl+A")
menu_edit.add_separator()
menu_edit.add_command(label = 'Copy Sonnet 1', command = lambda: copy_to_clipboard(1))
menu_edit.add_command(label = 'Copy Sonnet 2', command = lambda: copy_to_clipboard(2))
menu_edit.add_separator()
menu_edit.add_command(label = 'Clear Sonnet 1', command = lambda: clear_sonnet(1))
menu_edit.add_command(label = 'Clear Sonnet 2', command = lambda: clear_sonnet(2))
menu_edit.add_separator()
menu_edit.add_command(label = 'Clear Log', command = clear_log)
menu_about.add_command(label = 'About My Application', command = show_about_window)

# Bind shortcuts to functions
root.bind('<Control-w>', exit_program)
root.bind('<Control-z>', shortcut_undo)
root.bind('<Control-Shift-z>', shortcut_redo)
root.bind('<Control-x>', shortcut_cut)
root.bind('<Control-c>', shortcut_copy)
root.bind('<Control-v>', shortcut_paste)
root.bind('<Control-a>', shortcut_selectall)

# Frame for log
log_frame = ttk.Frame(main_frame, borderwidth = 8, relief = 'sunken', width = 46, height = 48)
log = Text(log_frame, state = 'disabled', width = 42, height = 48)
log.grid(column = 0, row = 1, pady = 12, sticky = 'n')
log.configure(font = 'times 12', background = 'gray26', foreground = 'snow')
# Initial log messages
write_to_log("Parsed 1711 lines" ,logger_index)
write_to_log("Using vocabulary size 2792" ,logger_index)
write_to_log("Using bigram size 13001" ,logger_index)

# Initial image of poem
sonnet_image = Image.open('calligraphy.jpg')
width_percent = base_width / float(sonnet_image.size[0])
height_size = int((float(sonnet_image.size[1]) * float(width_percent)))
display_sonnet_image = ImageTk.PhotoImage(sonnet_image.resize((base_width, height_size), Image.ANTIALIAS))

# Frame for poems
poem_frame = ttk.Frame(main_frame, width = 160, height = 60)
# Sonnet 1
ttk.Label(poem_frame, text = "Sonnet 1", font = 'helvetica 14 bold').grid(column = 0, row = 0)
poem1_frame = ttk.Frame(poem_frame)
poem1_frame.grid(column = 0, row = 1)
poem1_display = Text(poem1_frame, state = 'disabled', width = 64, height = 18)
poem1_display.grid(column = 0, row = 0, padx = 6, pady = 6)
poem1_display.configure(font = 'helvetica 14')
# Sonnet 1 image
poem1_image_frame = ttk.Frame(poem1_frame, width = 80, height = 16)
poem1_image_frame.grid(column = 1, row = 0, padx = 6, pady = 6)
poem1_image = ttk.Label(poem1_image_frame, image = display_sonnet_image)
poem1_image.pack(side = "bottom", fill = "both", expand = "yes")
# Separator
ttk.Separator(poem_frame, orient = HORIZONTAL).grid(column = 0, row = 2, pady = 24, sticky = 'ew')
# Sonnet 2
ttk.Label(poem_frame, text = "Sonnet 2", font = 'helvetica 14 bold').grid(column = 0, row = 3)
poem2_frame = ttk.Frame(poem_frame)
poem2_frame.grid(column = 0, row = 4)
poem2_display = Text(poem2_frame, state = 'disabled', width = 64, height = 18)
poem2_display.grid(column = 0, row = 0, padx = 6, pady = 6)
poem2_display.configure(font = 'helvetica 14')
# Sonnet 2 image
poem2_image_frame = ttk.Frame(poem2_frame, width = 80, height = 16)
poem2_image_frame.grid(column = 1, row = 0, padx = 6, pady = 6)
poem2_image = ttk.Label(poem2_image_frame, image = display_sonnet_image)
poem2_image.pack(side = "bottom", fill = "both", expand = "yes")

# Frame for buttons
button_frame = ttk.Frame(main_frame, width = 64, height = 60)

# Frame for models
model_frame = ttk.Frame(button_frame)
model_frame.grid(column = 0, row = 0)

# Current model
current_model = StringVar()
current_model.set("Current Model: None")
ttk.Label(model_frame, textvariable = current_model, font = 'helvetica 10 bold').grid(column = 0, row = 0, pady = 8)
# Number of iterations
ttk.Label(model_frame, text = "Number of iterations:").grid(column = 0, row = 1, sticky = 'nw', pady = (12, 4))
# Radio frame
radio_frame = ttk.Frame(model_frame)
radio_frame.grid(column = 0, row = 2, pady = 12)
modes = [("20", 20, 0), ("40", 40, 1), ("60", 60, 2), ("80", 80, 3), ("100", 100, 4)]
for text, number, i in modes:
  ttk.Radiobutton(radio_frame, text = text, variable = number_iterations, value = number).grid(column = i, row = 0)

# Load models
# Frame for buttons
load_buttons_frame = ttk.Frame(button_frame)
load_buttons_frame.grid(column = 0, row = 1, pady = 12)
# Load buttons
ttk.Button(load_buttons_frame, text = "Load RNN Model", command = load_rnn).grid(column = 0, row = 0, padx = 8, pady = 8)
ttk.Button(load_buttons_frame, text = "Load GRU Model", command = load_gru).grid(column = 1, row = 0, padx = 8, pady = 8)
ttk.Button(load_buttons_frame, text = "Load RNN Bigram", command = load_rnn_bigram).grid(column = 0, row = 1, padx = 8, pady = 8)
ttk.Button(load_buttons_frame, text = "Load GRU Bigram", command = load_gru_bigram).grid(column = 1, row = 1, padx = 8, pady = 8)
ttk.Button(load_buttons_frame, text = "Open Custom RNN", command = load_custom_rnn).grid(column = 0, row = 2, padx = 8, pady = 8)
ttk.Button(load_buttons_frame, text = "Open Custom GRU ", command = load_custom_gru).grid(column = 1, row = 2, padx = 8, pady = 8)
# Separator
ttk.Separator(button_frame, orient = HORIZONTAL).grid(column = 0, row = 2, pady = 16, sticky = 'ew')

# Sonnet 1
# Frame for sonnet 1
sonnet1_frame = ttk.Frame(button_frame)
sonnet1_frame.grid(column = 0, row = 3)
ttk.Label(sonnet1_frame, text = "Sonnet 1", font = 'helvetica 10 bold').grid(column = 0, row = 0, pady = 8)
# Theme
theme1_frame = ttk.Frame(sonnet1_frame)
theme1_frame.grid(column = 0, row = 1)
ttk.Label(theme1_frame, text = "Theme:").grid(column = 0, row = 0, sticky = 'nw', padx = 4, pady = 8)
theme1 = StringVar()
theme1_entry = ttk.Entry(theme1_frame, width = 24, textvariable = theme1).grid(column = 1, row = 0, padx = 4, pady = 8)
# Generate sonnet button
generate_sonnet1_button = ttk.Button(sonnet1_frame, text = "Generate Sonnet", state = 'disabled', command = lambda: generate_sonnet(1))
generate_sonnet1_button.grid(column = 0, row = 2, pady = 8)
generate_sonnet1_image_button = ttk.Button(sonnet1_frame, text = "Generate Sonnet based on Image", state = 'disabled', command = lambda: generate_sonnet_image(1))
generate_sonnet1_image_button.grid(column = 0, row = 3, pady = 8)
ttk.Button(sonnet1_frame, text = "Random Shakespeare's Sonnet", command = lambda: get_example_poem(1)).grid(column = 0, row = 4, pady = 8)
# BLEU Score
bleu1_frame = ttk.Frame(sonnet1_frame)
bleu1_frame.grid(column = 0, row = 5)
ttk.Button(bleu1_frame, text = "BLEU Score", command = lambda: bleu_score(1)).grid(column = 0, row = 0, padx = 8, pady = 8)
ttk.Button(bleu1_frame, text = "BLEU Score (n-gram)", command = lambda: bleu_ngram_score(1)).grid(column = 1, row = 0, padx = 8, pady = 8)
# Copy / Clear Sonnet
copy_clear1_frame = ttk.Frame(sonnet1_frame)
copy_clear1_frame.grid(column = 0, row = 6)
ttk.Button(copy_clear1_frame, text = "Copy to Clipboard", command = lambda: copy_to_clipboard(1)).grid(column = 0, row = 0, padx = 8, pady = 8)
ttk.Button(copy_clear1_frame, text = "Clear Sonnet", command = lambda: clear_sonnet(1)).grid(column = 1, row = 0, padx = 8, pady = 8)
# Separator
ttk.Separator(button_frame, orient = HORIZONTAL).grid(column = 0, row = 4, pady = 16, sticky = 'ew')

# Sonnet 2
# Frame for sonnet 2
sonnet2_frame = ttk.Frame(button_frame)
sonnet2_frame.grid(column = 0, row = 5)
ttk.Label(sonnet2_frame, text = "Sonnet 2", font = 'helvetica 10 bold').grid(column = 0, row = 0, pady = 8)
# Theme
theme2_frame = ttk.Frame(sonnet2_frame)
theme2_frame.grid(column = 0, row = 1)
ttk.Label(theme2_frame, text = "Theme:").grid(column = 0, row = 0, sticky = 'nw', padx = 4, pady = 8)
theme2 = StringVar()
theme2_entry = ttk.Entry(theme2_frame, width = 24, textvariable = theme2).grid(column = 1, row = 0, padx = 4, pady = 8)
# Generate sonnet button
generate_sonnet2_button = ttk.Button(sonnet2_frame, text = "Generate Sonnet", state = 'disabled', command = lambda: generate_sonnet(2))
generate_sonnet2_button.grid(column = 0, row = 2, pady = 8)
generate_sonnet2_image_button = ttk.Button(sonnet2_frame, text = "Generate Sonnet based on Image", state = 'disabled', command = lambda: generate_sonnet_image(2))
generate_sonnet2_image_button.grid(column = 0, row = 3, pady = 8)
ttk.Button(sonnet2_frame, text = "Random Shakespeare's Sonnet", command = lambda: get_example_poem(2)).grid(column = 0, row = 4, pady = 8)
# BLEU Score
bleu2_frame = ttk.Frame(sonnet2_frame)
bleu2_frame.grid(column = 0, row = 5)
ttk.Button(bleu2_frame, text = "BLEU Score", command = lambda: bleu_score(2)).grid(column = 0, row = 0, padx = 8, pady = 8)
ttk.Button(bleu2_frame, text = "BLEU Score (n-gram)", command = lambda: bleu_ngram_score(2)).grid(column = 1, row = 0, padx = 8, pady = 8)
# Copy / Clear Sonnet
copy_clear2_frame = ttk.Frame(sonnet2_frame)
copy_clear2_frame.grid(column = 0, row = 6)
ttk.Button(copy_clear2_frame, text = "Copy to Clipboard", command = lambda: copy_to_clipboard(2)).grid(column = 0, row = 0, padx = 8, pady = 8)
ttk.Button(copy_clear2_frame, text = "Clear Sonnet", command = lambda: clear_sonnet(2)).grid(column = 1, row = 0, padx = 8, pady = 8)
# Separator
ttk.Separator(button_frame, orient = HORIZONTAL).grid(column = 0, row = 6, pady = 16, sticky = 'ew')

# Others
ttk.Button(button_frame, text = "Clear Log", command = clear_log).grid(column = 0, row = 7, pady = 8)
ttk.Button(button_frame, text = "Exit", command = exit_program).grid(column = 0, row = 8, pady = 8)

# Outer frames coordinates
log_frame.grid(column = 0, row = 0, sticky = 'n', padx = 24, pady = 12)
poem_frame.grid(column = 1, row = 0, padx = 24, pady = 12)
button_frame.grid(column = 2, row = 0, sticky = 'ns', padx = 24, pady = 12)

root.mainloop()
