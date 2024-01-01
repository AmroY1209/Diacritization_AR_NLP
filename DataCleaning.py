import pandas as pd
import numpy as np
import re
from pyarabic.araby import strip_tashkeel


def replace_unicode_sequences(arabic_words: list) -> list:
        res = []
        for word in arabic_words:
            new_word = word
            if '\u0651\u064B' in new_word:
                new_word = (re.sub(r'\u0651\u064B', '١', new_word)) # shadda + tanween fatha
            if '\u0651\u064C' in new_word:
                new_word = (re.sub(r'\u0651\u064C', '٢', new_word))# shadda + tanween damma
            if '\u0651\u064D' in new_word:
                new_word = (re.sub(r'\u0651\u064D', '٣', new_word))# shadda + tanween kasra
            if '\u0651\u064E' in new_word:
                new_word = (re.sub(r'\u0651\u064E', '٤', new_word))# shadda + fatha
            if '\u0651\u064F' in new_word:
                new_word = (re.sub(r'\u0651\u064F', '٥', new_word))# shadda + damma
            if '\u0651\u0650' in new_word:
                new_word = (re.sub(r'\u0651\u0650', '٦', new_word))# shadda + kasra
            res.append(new_word)
        return res

def replace_unicode_sequences_in_sentence(sentence: str):
    words = sentence.split()
    replaced_words = replace_unicode_sequences(words)
    return ' '.join(replaced_words)


class DataCleaning:
    
    def __init__(self):
        self.train_words = []
        self.test_words = []
    
    
    def cleaning_training_data(self, filename = "train"):
        with open('./Dataset/training/' + filename + '.txt', 'r', encoding='utf-8') as file:
            train_txt = file.read()

        # train_words = []
        with open('./Dataset/training/' + filename + '_cleaned.txt', 'w', encoding='utf-8') as cleaned_file:
            with open('./Dataset/training/' + filename + '_stripped.txt', 'w', encoding='utf-8') as output_file:
                with open('./Dataset/training/' + filename + '_replace.txt', 'w', encoding='utf-8') as replace_file:
                    with open('./Dataset/training/' + filename + '_words.txt', 'w', encoding='utf-8') as words_file:
                        for sentence in train_txt.split('\n'):
                            sentence = re.sub(r'[^\u0600-\u0660 \.]+|[؛؟]', '', sentence)
                            sentence = re.sub(r' +', ' ', sentence)
                            sentence = sentence.strip()
                            cleaned_file.write(sentence + '\n')
                            output_file.write(strip_tashkeel(sentence) + '\n')
                            replace_file.write(replace_unicode_sequences_in_sentence(sentence) + '\n')
                            for word in sentence.split():
                                word = word.strip()
                                if word:
                                    self.train_words.append(word)
                                    words_file.write(word + '\n') 

        with open('./Dataset/training/train_words_replaced.txt', 'w', encoding='utf-8') as output_file:
            for word in replace_unicode_sequences(self.train_words):
                output_file.write(word + '\n')
                
    
    
    def cleaning_validation_data(self, filename = "test"):
        with open('./Dataset/test/' + filename + '.txt', 'r', encoding='utf-8') as file:
            test_txt = file.read()

        test_words = []
        with open('./Dataset/test/' + filename + '_cleaned.txt', 'w', encoding='utf-8') as cleaned_file:
            with open('./Dataset/test/' + filename + '_stripped.txt', 'w', encoding='utf-8') as output_file:
                with open('./Dataset/test/' + filename + '_replaced.txt', 'w', encoding='utf-8') as replaced_file:
                    with open('./Dataset/test/' + filename + '_words.txt', 'w', encoding='utf-8') as words_file:
                        for sentence in test_txt.split('\n'):
                            sentence = re.sub(r'[^\u0600-\u0660 \.]+|[؛؟]', '', sentence)
                            sentence = re.sub(r' +', ' ', sentence)
                            sentence = sentence.strip()
                            cleaned_file.write(sentence + '\n')
                            output_file.write(strip_tashkeel(sentence) + '\n')
                            replaced_file.write(replace_unicode_sequences_in_sentence(sentence) + '\n')
                            for word in sentence.split():
                                word = word.strip()
                                if word:
                                    self.test_words.append(word)
                                    words_file.write(word + '\n') 

        with open('./Dataset/test/test_words_replaced.txt', 'w', encoding='utf-8') as output_file:
            for word in replace_unicode_sequences(self.test_words):
                output_file.write(word + '\n')
                
                
    def strip_words(self, filename1 = "train", filename2 = "test"):
        train_words_stripped = list((strip_tashkeel(word) for word in self.train_words))
        with open('./Dataset/training/' + filename1 + '_words_stripped.txt', 'w', encoding='utf-8') as output_file:
            for word in train_words_stripped:
                output_file.write(word + '\n')
        test_words_stripped = list((strip_tashkeel(word) for word in self.test_words))
        with open('./Dataset/test/' + filename2 + '_words_stripped.txt', 'w', encoding='utf-8') as output_file:
            for word in test_words_stripped:
                output_file.write(word + '\n')