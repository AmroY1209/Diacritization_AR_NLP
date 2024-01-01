import pickle

with open('./pickles/test_sentence_diacritics_appearance.pickle', 'rb') as file:
    test_sentence_diacritics_appearance = pickle.load(file)
with open('./pickles/test_char_sequences_without_tashkeel.pkl', 'rb') as file:
    test_char_sequences_without_tashkeel_padded = pickle.load(file)
with open('./pickles/test_word_sequences.pkl', 'rb') as file:
    test_word_sequences = pickle.load(file)

with open('./pickles/test_tashkeel_sequences.pkl', 'rb') as file:
    test_tashkeel_sequences = pickle.load(file)

with open('./pickles/test_sentence_diacritics_appearance_sequences.pickle', 'rb') as file:
    test_sentence_diacritics_appearance_sequences_padded = pickle.load(file)

print(test_sentence_diacritics_appearance[0])
print("LOLOLOL")
print(test_char_sequences_without_tashkeel_padded[0])
print("LOLOLOL")
print(test_word_sequences.shape)
print("LOLOLOL")
print(test_tashkeel_sequences[0])
print("LOLOLOL")
print(test_sentence_diacritics_appearance_sequences_padded[0])
print("LOLOLOL")