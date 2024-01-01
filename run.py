from DataCleaning import DataCleaning
from Tokenization import Tokenization
from FeatureExtraction import FeatureExtraction
import pickle

# data_cleaning = DataCleaning()
# data_cleaning.cleaning_training_data()
# data_cleaning.cleaning_validation_data("test_no_diacritics")
# data_cleaning.strip_words()

ft = FeatureExtraction()
ft.load_dataset()
ft.load_segmented_sentences('training', 'train')
ft.load_segmented_sentences('test', 'test_no_diacritics')
dictionary = ft.get_letter_dictionary_from_file()
train_appearance = ft.get_sentence_diacritics_appearance("./Dataset/training/train_stripped.txt", 'sentence_diacritics_appearance')
test_appearance = ft.get_sentence_diacritics_appearance('./Dataset/test/test_no_diacritics_stripped.txt', 'test_sentence_diacritics_appearance')

print(len(test_appearance))

# tokenizer = Tokenization()
# tokenizer.load_data('train','test_no_diacritics')
# word_sequences_padded, char_sequences_without_tashkeel_padded, test_word_sequences_padded, test_char_sequences_without_tashkeel_padded = tokenizer.create_word_based_tokenizer()
# tokenizer.tashkeel_separation()
# tashkeel_list_sequences_padded, test_tashkeel_list_sequences_padded = tokenizer.tokenize_only_tashkeel()

# sentence_diacritics_appearance_sequences_padded, test_sentence_diacritics_appearance_sequences_padded = tokenizer.tokenize_diacritics_list()

# print(word_sequences_padded.shape)
# print(char_sequences_without_tashkeel_padded.shape)
# print(test_word_sequences_padded.shape)
# print(test_char_sequences_without_tashkeel_padded.shape)
# print(tashkeel_list_sequences_padded.shape)
# print(test_tashkeel_list_sequences_padded.shape)
# print(sentence_diacritics_appearance_sequences_padded.shape)
# print(test_sentence_diacritics_appearance_sequences_padded.shape)

