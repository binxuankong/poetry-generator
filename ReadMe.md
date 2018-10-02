# Poetry Generator using Deep Learning
## COMP30030 Third Year Project, University of Manchester, UK
### Created by: Bin Xuan Kong
### Supervised by: Dr. Tingting Mu

This program generates Shakespearean-styled sonnet using Recurrent Neural Network (RNN). A Shakespearen-styled sonnet has
- themes of passage of time, love beauty, mortality
- 14 lines with around 10 syllables each line
- a rhyming scheme of *abab cdcd efef gg*.

## Setup

Install Python requirements
```
pip install -r requirements.txt
```

## Running the program

To run the program without any GUI
```
python PoetryGenerator.py -m *mode* -i *iteratons* -t *theme*
```
- mode is the type of neural network model to use, with a selection of RNN, RNNBigram, GRU, GRUBigram
- iteration is the number of iterations trained of the model, with a selection of 20, 40, 60, 80, 100
- theme is the theme of the generated poem, is no theme is given a random poem will be generated

<br />
To run the program with GUI
For Python 2.7
```
python PoetryGeneratorV2.py
```
For Python 3.4 and above
```
python PoetryGeneratorV3.py
```
The functionality of the program with GUI includes
- selection of mode and iteration
- loading own pretrained model
- generating a poem based on a theme or image
- getting examples of Shakespeare's sonnets

<br />
To train your own neural netowrk model
```
python TrainModel.py -m *mode* -i *iteratons* -f *filename*
```
- mode is the type of neural network model to use, with a selection of RNN, RNNBigram, GRU, GRUBigram
- iteration is the number of iterations to train the model
- filename is the name to save the file as, must be of *.npz format

Note that the trained models are not included as the file size is too big to be uploaded. Please train your own model for the program to be functionable.
