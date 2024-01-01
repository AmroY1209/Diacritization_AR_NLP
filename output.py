import pickle
import torch
from torch import nn
import csv

with open("./pickles/tashkeel_index.pkl", "rb") as file:
    tashkeel_index = pickle.load(file)

with open("./pickles/diacritic2id.pickle", "rb") as file:
    tashkeel_index_official = pickle.load(file)

with open("./pickles/test_output.pkl", "rb") as file:
    output = pickle.load(file)

softmax_output = nn.functional.softmax(output, dim=-1)

# Find the index of the maximum value along the last axis
max_arg = torch.argmax(softmax_output, dim=-1)

# Add a new dimension at the end to make the shape (2000, 1904, 1)
new_tensor = torch.unsqueeze(max_arg, dim=-1)

print(new_tensor.shape)


# Golden
## { َ  : 0, ً : 1, ُ : 2, ٌ : 3, ِ  : 4, ٍ  : 5, ْ : 6, ّ  : 7, ّ َ  : 8, ّ ً : 9, ّ ُ : 10, ّ ٌ : 11, ّ ِ  : 12,  ّ ٍ : 13, '': 14}

# Define the mapping dictionary
mapping_dict = {
    0: 14,
    1: 14,
    2: 14,
    3: 0,
    4: 4,
    5: 6,
    6: 2,
    7: 8,
    8: 5,
    9: 12,
    10: 1,
    11: 3,
    12: 10,
    13: 7,
    14: 13,
    15: 11,
    16: 9
}

print(new_tensor[0][:10])
# Apply the mapping using vectorized operations
# If a key is not found in the mapping dictionary, use a default value of -1
new_tensor = torch.tensor([[mapping_dict.get(elem.item(), -1) for elem in row] for row in new_tensor], dtype=torch.long)

# Replace any occurrences of -1 with a default value (you can adjust this value)
default_value = 0
new_tensor[new_tensor == -1] = default_value

print(new_tensor[0])

# generate csv file
with open('./Dataset/test/test_no_diacritics_stripped.txt', 'r', encoding='utf-8') as file:
    test_txt = file.readlines()
        
list_of_sentences = []
for sentence in test_txt:
    list_of_sentences.append(sentence.strip())

# Create a list of lists with an added ID column and a single label column
csv_data = [['id', 'label']]

row = 0
column = 0
id = 0
for sentence in list_of_sentences:
    for char in sentence:
        if char == " " or char == ".":
            column += 1
            continue

        csv_data.append([id, new_tensor[row][column].item()])

        id += 1
        column += 1

    row += 1
    column = 0

with open("answer.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f'CSV file has been created.')