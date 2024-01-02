# %%
import torch

import pickle

import numpy as np

from tqdm import tqdm

from torch import nn

# Encodes categorical labels into numerical format (used for label preprocessing)
from sklearn.preprocessing import LabelEncoder

# Calculates the accuracy of a classification model (used for model evaluation)
from sklearn.metrics import accuracy_score

# Defines a custom dataset class for PyTorch (used for handling data)
import torch.utils.data

# Creates a DataLoader for efficient batch processing in PyTorch (used for data loading)
from torch.utils.data import DataLoader

# Splits a dataset into training and validation sets (used for data splitting)
from torch.utils.data import random_split

# Represents a multi-dimensional matrix in PyTorch (used for tensor manipulation)
from torch import Tensor

# Implements a linear layer in a neural network (used for defining neural network architecture)
from torch.nn import Linear

# Applies rectified linear unit (ReLU) activation function (used for introducing non-linearity)
from torch.nn import ReLU

# Applies sigmoid activation function (used for binary classification output)
from torch.nn import Sigmoid

# Base class for all neural network modules in PyTorch (used for creating custom models)
from torch.nn import Module

# Stochastic Gradient Descent optimizer (used for model optimization during training)
from torch.optim import SGD

# Binary Cross Entropy Loss function (used for binary classification problems)
from torch.nn import BCELoss

# Initializes weights using Kaiming uniform initialization (used for weight initialization)
from torch.nn.init import kaiming_uniform_

# Initializes weights using Xavier (Glorot) uniform initialization (used for weight initialization)
from torch.nn.init import xavier_uniform_

from torch.nn.utils.rnn import pad_sequence

# %%

with open('./pickles/word_sequences.pkl', 'rb') as file:
    word_sequences = pickle.load(file)

with open('./pickles/char_sequences_without_tashkeel.pkl', 'rb') as file:
    char_sequences = pickle.load(file)

with open('./pickles/tashkeel_sequences.pkl', 'rb') as file:
    labels = pickle.load(file)

with open('./pickles/val_word_sequences.pkl', 'rb') as file:
    val_word_sequences = pickle.load(file)

with open('./pickles/val_char_sequences_without_tashkeel.pkl', 'rb') as file:
    val_char_sequences = pickle.load(file)

with open('./pickles/val_tashkeel_sequences.pkl', 'rb') as file:
    val_labels = pickle.load(file)

with open('./pickles/sentence_diacritics_appearance_sequences.pickle', 'rb') as file:
    test_sentences_diacritics_sequences = pickle.load(file)

with open('./pickles/val_sentence_diacritics_appearance_sequences.pickle', 'rb') as file:
    val_sentences_diacritics_sequences = pickle.load(file)

with open('./pickles/segment_sequences.pickle', 'rb') as file:
    train_segment_sequences = pickle.load(file)

with open('./pickles/val_segment_sequences.pickle', 'rb') as file:
    val_segment_sequences = pickle.load(file)

# %%
print(len(word_sequences))
print(len(char_sequences[1]))
print(len(labels[1]))
print(len(test_sentences_diacritics_sequences[0]))

# %% [markdown]
# # Utility functions

# %%
def concatenate_characters(characters):
    # Create a tensor of zeros with the same shape as the last subsequence
    zeros_tensor = torch.zeros_like(characters[:, 0:1, :])

    # Concatenate it to the original tensor along the second dimension
    padded_x = torch.cat((characters, zeros_tensor), dim=1)

    # Now, padded_x will have zeros padded to the last subsequence

    temp1 = padded_x[:, :-1, :]
    temp2 = padded_x[:, 1:, :]

    # Concatenate along the last dimension
    concatenated_characters = torch.cat((temp1, temp2), dim=-1)

    return concatenated_characters

def concatenate_characters2(characters):
    # Create a tensor of zeros with the same shape as the last subsequence
    zeros_tensor = torch.zeros_like(characters[:, 0:1])

    # Concatenate it to the original tensor along the second dimension
    padded_x = torch.cat((characters, zeros_tensor), dim=1)

    # Now, padded_x will have zeros padded to the last subsequence

    temp1 = padded_x[:, :-1]
    temp2 = padded_x[:, 1:]

    # Concatenate along the last dimension
    concatenated_characters = torch.cat((temp1, temp2), dim=-1)

    return concatenated_characters

def concatenate_tensors_elementwise(tensor1, tensor2):
    result = torch.cat((tensor1, tensor2), dim=-1)
    return result

def concatenate_tensors_feature3(tensor1, tensor2):
    concatenated_tensor = torch.cat((tensor1, tensor2.unsqueeze(2)), dim=2)
    return concatenated_tensor

def pad_list(list, max_len, val):
    for i in range(len(list)):
        if len(list[i]) < max_len:
            for j in range(max_len - len(list[i])):
                list[i].append(val)

    return list

# %%
# dataset definition
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
class Dataset(torch.utils.data.Dataset):
    # load the dataset
    # The __init__ function is run once when instantiating the Dataset object
    def __init__(self, char_sequences, labels, word_sequences, diacritics_sequence, segment_sequences):
        
        self.x = torch.tensor(char_sequences)

        self.y = torch.tensor(labels)

        self.word = torch.tensor(word_sequences)

        self.diacritics_sequence = torch.tensor(diacritics_sequence)

        self.segment_sequences = torch.tensor(segment_sequences)

    # number of rows in the dataset
    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.word[idx], self.diacritics_sequence[idx], self.segment_sequences[idx]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.x))
        train_size = len(self.x) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# # prepare the dataset
# def prepare_data():
#     # load the dataset
#     dataset = CSVDataset()
#     # calculate split
#     train, test = dataset.get_splits()
#     # prepare data loaders
#     # The Dataset retrieves our dataset’s features and labels one sample at a time.
#     # While training a model, we typically want to pass samples in “minibatches”,
#     # reshuffle the data at every epoch to reduce model overfitting,
#     train_dl = DataLoader(train, batch_size=32, shuffle=True)
#     test_dl = DataLoader(test, batch_size=1024, shuffle=False)
#     return dataset.encoding_mapping, train_dl, test_dl

# %%
#convert labels to numpy array
print(len(labels[3]))
print(len(char_sequences[3]))
train_ds = Dataset(char_sequences, labels, word_sequences, test_sentences_diacritics_sequences, train_segment_sequences)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

dg = iter(train_dl)
X1, Y1, z1, d1, s1 = next(dg)
X2, Y2, z2, d2, s2 = next(dg)
print(Y1.shape, X1.shape, z1.shape, d1.shape, s1.shape, Y2.shape, X2.shape, z2.shape, d2.shape, s2.shape)
print(X1[0][:], "\n", Y1[0][:])

# %% [markdown]
# MODEL

# %%
class Char_model(nn.Module):
  def __init__(self, vocab_size=42, embedding_dim=50, hidden_size=50, n_classes=17):
    """
    The constructor of our NER model
    Inputs:
    - vacab_size: the number of unique words
    - embedding_dim: the embedding dimension
    - n_classes: the number of final classes (tags)

    embedding_dim here: 50 for char embedding + 50 for following char embedding + 1 for feature3 = 101
    """

    super(Char_model, self).__init__()

    input_len = 2*embedding_dim + 15 + 1
    ####################### TODO: Create the layers of your model #######################################
    # (1) Create the embedding layer
    self.embedding_char = nn.Embedding(vocab_size, embedding_dim)
    self.embedding_diacritics = nn.Embedding(14, 15)

    # (2) Create an LSTM layer with hidden size = hidden_size and batch_first = True
    self.lstm = nn.LSTM(input_len, hidden_size, batch_first=True)
    # batch_first makes the input and output tensors to be of shape (batch_size, seq_length, hidden_size)

    # (3) Create a linear layer
    self.linear = nn.Linear(hidden_size, n_classes)

    #####################################################################################################

  def forward(self, sentences, diacritics_list, segments, h_0=None, c_0=None):
    """
    This function does the forward pass of our model
    Inputs:
    - sentences: tensor of shape (batch_size, max_length)

    Returns:
    - final_output: tensor of shape (batch_size, max_length, n_classes)
    """

    final_output = None
    #############################################################
    sentences_embedded = self.embedding_char(sentences) 
    diacritics_embedded = self.embedding_diacritics(diacritics_list)
    
    sentences_embedded = concatenate_characters(sentences_embedded) #feature 1: concatenate characters
    sentences_embedded = concatenate_tensors_elementwise(sentences_embedded, diacritics_embedded) #feature 2: concatenate diacritics seen before
    sentences_embedded = concatenate_tensors_feature3(sentences_embedded, segments) #feature 3: concatenate segment for each character

    #check if h_0 and c_0 are provided or not
    if h_0 is None or c_0 is None:
      final_output, (h_0, c_0) = self.lstm(sentences_embedded)
    else:
      final_output, _ = self.lstm(sentences_embedded, (h_0, c_0)) 
      
    final_output = self.linear(final_output)  


    ############################################################
    return final_output

# %%
class Word_model(nn.Module):
  def __init__(self, vocab_size=2093761, embedding_dim=50, hidden_size=50, n_classes=17):
    """
    The constructor of our NER model
    Inputs:
    - vacab_size: the number of unique words
    - embedding_dim: the embedding dimension
    - n_classes: the number of final classes (tags)
    """
    super(Word_model, self).__init__()
    ####################### TODO: Create the layers of your model #######################################
    # (1) Create the embedding layer
    self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # (2) Create an LSTM layer with hidden size = hidden_size and batch_first = True
    self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
    # batch_first makes the input and output tensors to be of shape (batch_size, seq_length, hidden_size)

    #####################################################################################################

  def forward(self, sentences):
    """
    This function does the forward pass of our model
    Inputs:
    - sentences: tensor of shape (batch_size, max_length)

    Returns:
    - final_output: tensor of shape (batch_size, max_length, n_classes)
    """

    final_output = None
    ######################### TODO: implement the forward pass ####################################
    final_output = self.embedding(sentences) 
    final_output, h = self.rnn(final_output)   

    ###############################################################################################
    return final_output, h

# %%
lstm_model = Char_model()
word_model = Word_model()
# lstm_model.load_state_dict(torch.load('lstm_model_weights.pth'))
# word_model.load_state_dict(torch.load('word_model_weights.pth'))
print(lstm_model)
print(word_model)

# %% [markdown]
# # Training

# %%
def train(lstm_model, context_model, train_dataset, batch_size=128, epochs=22, learning_rate=0.014):
  """
  This function implements the training logic
  Inputs:
  - model: the model ot be trained
  - train_dataset: the training set of type NERDataset
  - batch_size: integer represents the number of examples per step
  - epochs: integer represents the total number of epochs (full training pass)
  - learning_rate: the learning rate to be used by the optimizer
  """

  # (1) create the dataloader of the training set (make the shuffle=True)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  # (2) make the criterion cross entropy loss
  criterion = nn.CrossEntropyLoss()

  # (3) create the optimizer (Adam)
  optimizer = torch.optim.Adam(list(lstm_model.parameters()) + list(context_model.parameters()), lr=learning_rate)
  # GPU configuration
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  if use_cuda:
    lstm_model = lstm_model.cuda()
    context_model = context_model.cuda()
    criterion = criterion.cuda()
  # device="cpu"
  for epoch_num in range(epochs):
    total_acc_train = 0
    total_loss_train = 0

    for train_input, train_label, train_context, train_diacritic, train_segments in tqdm(train_dataloader):
      

      # (4) move the train input to the device
      train_label = train_label.long().to(device)

      # (5) move the train label to the device
      train_input = train_input.long().to(device)

      train_context = train_context.long().to(device)

      train_diacritic = train_diacritic.long().to(device)

      train_segments = train_segments.long().to(device)

      # (6) do the forward pass
      # context, h_0 = context_model(train_context)
      # c_0 = torch.zeros(context.shape[0], 1, context.shape[2])
      # h_0 = torch.transpose(h_0, 0, 1)
      #h_0 = h_0.permute(1, 0, 2)
      #print(h_0.shape)
      # print(train_input.shape)
      # print(train_diacritic.shape)
      output = lstm_model(train_input, train_diacritic, train_segments)
      
      # (7) loss calculation (you need to think in this part how to calculate the loss correctly)
      batch_loss = criterion(output.reshape(-1, 17), train_label.reshape(-1))
  
      # (8) append the batch loss to the total_loss_train
      total_loss_train += batch_loss.item()
      
      # (9) calculate the batch accuracy (just add the number of correct predictions)
      acc = (output.argmax(dim=2) == train_label).sum().item()
      total_acc_train += acc

      # (10) zero your gradients
      optimizer.zero_grad()
      
      # (11) do the backward pass
      batch_loss.backward()

      # (12) update the weights with your optimizer
      optimizer.step()
      
    # epoch loss
    epoch_loss = total_loss_train / len(train_dataset)

    # (13) calculate the accuracy
    epoch_acc = total_acc_train / (len(train_dataset) * len(train_dataset[0][0]))
    # ba2sem 3la 3adad el kalemat fy kol el gomal 
    # kol gomla asln fyha 104 kelma, fa badrab dh fy 3adad el gomal bs

    print(
        f'Epochs: {epoch_num + 1} | Train Loss: {epoch_loss} \
        | Train Accuracy: {epoch_acc}\n')

  ##############################################################################################################
  

# %%
torch.cuda.empty_cache()

train_dataset = Dataset(char_sequences, labels, word_sequences, test_sentences_diacritics_sequences, train_segment_sequences)
train(lstm_model, word_model, train_dataset)

# %%
torch.save(lstm_model.state_dict(), '30e5_lstm_model_weights.pth')
torch.save(word_model.state_dict(), '30e5_word_model_weights.pth')
tensor2 = tensor1 = torch.tensor([
    [[1, 2, 3], [4, 5, 6], [66, 55, 77]], 
    [[7, 8, 9], [10, 11, 12], [13, 14, 15]]
])

x = torch.tensor([
    [[1, 2, 3], [4, 5, 6], [66, 55, 77]], 
    [[7, 8, 9], [10, 11, 12], [13, 14, 15]]
])

# Get the shape of the input tensor
batch_size, sequence_length, feature_size = x.shape

# Create a tensor of zeros with the same shape as the last subsequence
zeros_tensor = torch.zeros_like(x[:, 0:1, :])

# Concatenate it to the original tensor along the second dimension
padded_x = torch.cat((x, zeros_tensor), dim=1)

# Now, padded_x will have zeros padded to the last subsequence
print(padded_x)

# %%
tensor1 = torch.tensor([
    [[1, 2, 3], [4, 5, 6], [66, 55, 77]], 
    [[7, 8, 9], [10, 11, 12], [13, 14, 15]]
])

# Concatenate tensor1 with itself along the last dimension
result = torch.cat((tensor1, tensor1), dim=-1)

print(result)

# %%
# List of strings
list_of_strings = ["hello", "world", "deep", "learning"]

# Convert strings to lists of character indices
list_of_lists = [list(map(ord, s)) for s in list_of_strings]

# Pad sequences to the same length
padded_sequences = pad_sequence([torch.tensor(seq) for seq in list_of_lists], batch_first=True, padding_value=0)

print(padded_sequences)

# %%
# Assuming you have two tensors A and B
tensor_A = torch.randn(128, 7183, 50)
tensor_B = torch.randint(0, 2, (128, 7183), dtype=torch.float32)  # Example tensor, adjust as needed

# Concatenate along the last dimension
concatenated_tensor = torch.cat((tensor_A, tensor_B.unsqueeze(2)), dim=2)

# Print the shape of the concatenated tensor
print(concatenated_tensor.shape)

# %% [markdown]
# # Evaluation

# %%
def evaluate(model, test_dataset, batch_size=256):
  """
  This function takes a NER model and evaluates its performance (accuracy) on a test data
  Inputs:
  - model: a NER model
  - test_dataset: dataset of type NERDataset
  """
  ########################### TODO: Replace the Nones in the following code ##########################

  # (1) create the test data loader
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

  # GPU Configuration
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  if use_cuda:
    model = model.cuda()

  total_acc_test = 0
  
  # (2) disable gradients
  with torch.no_grad():
    # 3mlna disable 3lshan e7na bn-predict (aw evaluate y3ny) b2a dlw2ty, msh bn-train

    for test_input, test_label, test_context, test_diacritics, test_segments in tqdm(test_dataloader):
      # (3) move the test input to the device
      test_label = test_label.to(device)

      # (4) move the test label to the device
      test_input = test_input.to(device)
      # brdo the comments should be reversed 
      test_context = test_context.long().to(device)
      test_diacritics = test_diacritics.long().to(device)
      test_segments = test_segments.long().to(device)
      # (5) do the forward pass
      output = model(test_input, test_diacritics, test_segments)
      print(test_input.shape)
      print(test_label.shape)
      print(output.shape)
      # accuracy calculation (just add the correct predicted items to total_acc_test)
      acc = (output.argmax(dim=2) == test_label).sum().item()
      total_acc_test += acc
    
    # (6) calculate the over all accuracy
    total_acc_test /= (len(test_dataset) * len(test_dataset[0][0]))
  ##################################################################################################

  
  print(f'\nTest Accuracy: {total_acc_test}')

# %%
test_dataset = Dataset(val_char_sequences, val_labels, val_word_sequences, val_sentences_diacritics_sequences, val_segment_sequences)
evaluate(lstm_model, test_dataset)

# %%
with open('./pickles/test_segment_sequences.pickle', 'rb') as file:
    test_segment_sequences = pickle.load(file)

with open('./pickles/test_sentence_diacritics_appearance_sequences.pickle', 'rb') as file:
    test_sentences_diacritics_sequences = pickle.load(file)

with open('./pickles/test_char_sequences_without_tashkeel.pkl', 'rb') as file:
    test_char_sequences = pickle.load(file)

print(len(test_segment_sequences[0]))
print(len(test_sentences_diacritics_sequences[0]))
print(len(test_char_sequences[0]))


# %%

torch.cuda.empty_cache()

lstm_model.load_state_dict(torch.load('30e5_lstm_model_weights.pth'))


output = lstm_model(torch.tensor(test_char_sequences), torch.tensor(test_sentences_diacritics_sequences), torch.tensor(test_segment_sequences))

print(output.shape)


# %% [markdown]
# # Extracting max Probabilities
# 

# %%
outputs = pickle.load(open("./pickles/test_output.pkl", "rb"))

print(outputs[0][0])

softmax_output = nn.functional.softmax(outputs, dim=-1)
print(softmax_output[0][0])

# Find the index of the maximum value along the last axis
max_arg = torch.argmax(softmax_output, dim=-1)

# Add a new dimension at the end to make the shape (2000, 1904, 1)
new_tensor = torch.unsqueeze(max_arg, dim=-1)

print(new_tensor[0][0])


