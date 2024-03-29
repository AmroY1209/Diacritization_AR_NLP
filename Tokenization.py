import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import nltk
import pyarabic.araby as araby
from pyarabic.araby import strip_tashkeel
import qalsadi.lemmatizer 
import qalsadi.analex as qa
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from farasa.pos import FarasaPOSTagger 
from farasa.ner import FarasaNamedEntityRecognizer 
from farasa.diacratizer import FarasaDiacritizer 
from farasa.segmenter import FarasaSegmenter 
from farasa.stemmer import FarasaStemmer
import keras
from diacritization_evaluation import util



class Tokenization:

    def __init__(self):
        self.words = []
        self.sentences = []
        self.sentences_with_tashkeel = []

        self.test_words = []
        self.test_sentences = []
        self.test_sentences_with_tashkeel = []

        self.sentences_replaced = []
        self.test_sentences_replaced = []


        self.tashkeel_list = []
        self.test_tashkeel_list = []

        self.sentence_diacritics_appearance = {}
        self.test_sentence_diacritics_appearance = {}

        self.sentences_segments = []
        self.test_sentences_segments = []



        
    def load_data(self,filename = "train",testfilename="test"):
        with open('./Dataset/training/'+filename+'_words_stripped.txt', 'r', encoding='utf-8') as output_file:
            for word in output_file:
                self.words.append(word.strip())

        with open('./Dataset/training/'+filename+'_stripped.txt', 'r', encoding='utf-8') as output_file:
            for sentence in output_file:
                self.sentences.append(sentence.strip())

        with open('./Dataset/training/'+filename+'_cleaned.txt', 'r', encoding='utf-8') as output_file:
            for sentence in output_file:
                self.sentences_with_tashkeel.append(sentence.strip())
        with open('./Dataset/training/'+filename+'_replace.txt', 'r', encoding='utf-8') as output_file:
            for sentence in output_file:
                self.sentences_replaced.append(sentence.strip())

        

        with open('./Dataset/test/test_words_stripped.txt', 'r', encoding='utf-8') as output_file:
            for word in output_file:
                self.test_words.append(word.strip())

        with open('./Dataset/test/'+testfilename+'_stripped.txt', 'r', encoding='utf-8') as output_file:
            for sentence in output_file:
                self.test_sentences.append(sentence.strip())

        with open('./Dataset/test/'+testfilename+'_cleaned.txt', 'r', encoding='utf-8') as output_file:
            for sentence in output_file:
                self.test_sentences_with_tashkeel.append(sentence.strip())

        with open('./Dataset/test/'+testfilename+'_replaced.txt', 'r', encoding='utf-8') as output_file:
            for sentence in output_file:
                self.test_sentences_replaced.append(sentence.strip())

        with open('./pickles/sentence_diacritics_appearance.pickle', 'rb') as file:
            self.sentence_diacritics_appearance = pickle.load(file)
        
        with open('./pickles/test_sentence_diacritics_appearance.pickle', 'rb') as file:
            self.test_sentence_diacritics_appearance = pickle.load(file)

        with open('./Dataset/training/'+filename+'_segmented.txt', 'r', encoding='utf-8') as output_file:
            for sentence in output_file:
                self.sentences_segments.append(sentence.strip())

        with open('./Dataset/test/'+testfilename+'_segmented.txt', 'r', encoding='utf-8') as output_file:
            for sentence in output_file:
                self.test_sentences_segments.append(sentence.strip())



        
    def create_word_based_tokenizer(self):
        words_tokenizer = Tokenizer()

        # Fit the tokenizer on the list of words (treat each word as a separate "sentence")
        words_tokenizer.fit_on_texts(self.sentences)

        # Get the word index
        word_index = words_tokenizer.word_index

        # Tokenize the words
        test_word_sequences = words_tokenizer.texts_to_sequences(self.test_sentences)


        char_tokenizer_without_tashkeel = Tokenizer(char_level=True)
        char_tokenizer_without_tashkeel.fit_on_texts(self.sentences)

        char_sequences_without_tashkeel = char_tokenizer_without_tashkeel.texts_to_sequences(self.sentences)
        test_char_sequences_without_tashkeel = char_tokenizer_without_tashkeel.texts_to_sequences(self.test_sentences)

        # Assuming word_sequences and char_sequences are the output of the tokenizers
        word_sequences = words_tokenizer.texts_to_sequences(self.sentences)
        # Add padding
        word_sequences_padded = pad_sequences(word_sequences, padding='post')

        char_sequences_without_tashkeel_padded = pad_sequences(char_sequences_without_tashkeel, padding='post')

        test_word_sequences_padded = pad_sequences(test_word_sequences, padding='post')

        test_char_sequences_without_tashkeel_padded = pad_sequences(test_char_sequences_without_tashkeel, padding='post')

        with open('./pickles/word_sequences.pkl', 'wb') as file:
            pickle.dump(word_sequences_padded, file)

        with open('./pickles/char_sequences_without_tashkeel.pkl', 'wb') as file:
            pickle.dump(char_sequences_without_tashkeel_padded, file)

        with open('./pickles/test_word_sequences.pkl', 'wb') as file:
            pickle.dump(test_word_sequences_padded, file)

        with open('./pickles/test_char_sequences_without_tashkeel.pkl', 'wb') as file:
            pickle.dump(test_char_sequences_without_tashkeel_padded, file)

        return word_sequences_padded, char_sequences_without_tashkeel_padded, test_word_sequences_padded, test_char_sequences_without_tashkeel_padded

      
    def tashkeel_separation(self):
        for sentence in self.sentences_with_tashkeel:
            _, _, harakat_list = util.extract_haraqat(sentence)   
            for i in range(len(harakat_list)):
                if len(harakat_list[i]) == 2:
                    if '\u0651\u064B' in harakat_list[i]:
                        harakat_list[i] = '١'
                    if '\u0651\u064C' in harakat_list[i]:
                        harakat_list[i] = '٢'
                    if '\u0651\u064D' in harakat_list[i]:
                        harakat_list[i] = '٣'
                    if '\u0651\u064E' in harakat_list[i]:
                        harakat_list[i] = '٤'
                    if '\u0651\u064F' in harakat_list[i]:
                        harakat_list[i] = '٥'
                    if '\u0651\u0650' in harakat_list[i]:
                        harakat_list[i] = '٦'

            self.tashkeel_list.append(harakat_list)

        for sentence in self.test_sentences_with_tashkeel:
            _, _, harakat_list = util.extract_haraqat(sentence)   
            for i in range(len(harakat_list)):
                if len(harakat_list[i]) == 2:
                    if '\u0651\u064B' in harakat_list[i]:
                        harakat_list[i] = '١'
                    if '\u0651\u064C' in harakat_list[i]:
                        harakat_list[i] = '٢'
                    if '\u0651\u064D' in harakat_list[i]:
                        harakat_list[i] = '٣'
                    if '\u0651\u064E' in harakat_list[i]:
                        harakat_list[i] = '٤'
                    if '\u0651\u064F' in harakat_list[i]:
                        harakat_list[i] = '٥'
                    if '\u0651\u0650' in harakat_list[i]:
                        harakat_list[i] = '٦'

            self.test_tashkeel_list.append(harakat_list)
            

      
    def tokenize_only_tashkeel(self):
        tashkeel_tokenizer = Tokenizer(char_level=True, oov_token='UNK')
        tashkeel_tokenizer.fit_on_texts(self.tashkeel_list)

        tashkeel_list_sequences = tashkeel_tokenizer.texts_to_sequences(self.tashkeel_list)
        test_tashkeel_list_sequences = tashkeel_tokenizer.texts_to_sequences(self.test_tashkeel_list)
        
        tashkeel_list_sequences_padded = pad_sequences(tashkeel_list_sequences, padding='post')
        test_tashkeel_list_sequences_padded = pad_sequences(test_tashkeel_list_sequences, padding='post')
        with open('./pickles/tashkeel_sequences.pkl', 'wb') as file:
            pickle.dump(tashkeel_list_sequences_padded, file)

        with open('./pickles/test_tashkeel_sequences.pkl', 'wb') as file:
            pickle.dump(test_tashkeel_list_sequences_padded, file)

        return tashkeel_list_sequences_padded, test_tashkeel_list_sequences_padded


    def tokenize_diacritics_list(self):
        sentence_diacritics_appearance_tokenizer = Tokenizer(oov_token='UNK')
        sentence_diacritics_appearance_tokenizer.fit_on_texts(self.sentence_diacritics_appearance)
        sentence_diacritics_appearance_sequences = sentence_diacritics_appearance_tokenizer.texts_to_sequences(self.sentence_diacritics_appearance)

        sentence_diacritics_appearance_sequences_padded = pad_sequences(sentence_diacritics_appearance_sequences, padding='post')

        test_sentence_diacritics_appearance_sequences = Tokenizer(oov_token='UNK')
        test_sentence_diacritics_appearance_sequences.fit_on_texts(self.test_sentence_diacritics_appearance)
        test_sentence_diacritics_appearance_sequences = sentence_diacritics_appearance_tokenizer.texts_to_sequences(self.test_sentence_diacritics_appearance)

        test_sentence_diacritics_appearance_sequences_padded = pad_sequences(test_sentence_diacritics_appearance_sequences, padding='post')

        with open('./pickles/sentence_diacritics_appearance_sequences.pickle', 'wb') as file:
            pickle.dump(sentence_diacritics_appearance_sequences_padded, file)

        with open('./pickles/test_sentence_diacritics_appearance_sequences.pickle', 'wb') as file:
            pickle.dump(test_sentence_diacritics_appearance_sequences_padded, file)

        return sentence_diacritics_appearance_sequences_padded, test_sentence_diacritics_appearance_sequences_padded

    def tokenize_segments(self):
        segment_tokenizer = Tokenizer(char_level= True, oov_token='UNK')
        segment_tokenizer.fit_on_texts(self.sentences_segments)
        segment_sequences = segment_tokenizer.texts_to_sequences(self.sentences_segments)

        segment_sequences_padded = pad_sequences(segment_sequences, padding='post')

        test_segment_sequences = Tokenizer(char_level= True, oov_token='UNK')
        test_segment_sequences.fit_on_texts(self.test_sentences_segments)
        test_segment_sequences = segment_tokenizer.texts_to_sequences(self.test_sentences_segments)

        test_segment_sequences_padded = pad_sequences(test_segment_sequences, padding='post')
        
        with open('./pickles/segment_sequences.pickle', 'wb') as file:
            pickle.dump(segment_sequences_padded, file)

        with open('./pickles/test_segment_sequences.pickle', 'wb') as file:
            pickle.dump(test_segment_sequences_padded, file)

        return segment_sequences_padded, test_segment_sequences_padded


    
