from DataCleaning import DataCleaning
from Tokenization import Tokenization



data_cleaning = DataCleaning()
data_cleaning.cleaning_training_data()
data_cleaning.cleaning_validation_data()
data_cleaning.strip_words()

tokenizer = Tokenization()
tokenizer.load_data('train','test')
word_sequences_padded, char_sequences_without_tashkeel_padded, test_word_sequences_padded, test_char_sequences_without_tashkeel_padded = tokenizer.create_word_based_tokenizer()
tokenizer.tashkeel_separation()
tashkeel_list_sequences_padded, test_tashkeel_list_sequences_padded = tokenizer.tokenize_only_tashkeel()

print(word_sequences_padded.shape)
print(char_sequences_without_tashkeel_padded.shape)
print(test_word_sequences_padded.shape)
print(test_char_sequences_without_tashkeel_padded.shape)
print(tashkeel_list_sequences_padded.shape)
print(test_tashkeel_list_sequences_padded.shape)


