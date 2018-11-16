import os
import re

def clean(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Originally taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_and_split(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Originally taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()

def to_word_list(filename):
    for line in filename:
        return clean_and_split(line)
    
def file_to_word_list(filename):
    words = []
    for line in filename:
        words.append(line.strip())
    return words

def file_to_string(filename):
    for line in filename:
        return line
    
def load_data_files(folder):
    labels = []
    data = []
    for f in os.listdir("data/" + folder):
        first = 0 if f[0] == "d" else 1 
        second = 0 if folder[:3] == "neg" else 1
        labels.append((first, second))
        text = file_to_string(open("data/" + folder + "/" + f))
        data.append(text)
    return labels, data

def load_data(folder):
    labels = []
    data = []
    for f in os.listdir("data/" + folder):
        first = 0 if f[0] == "d" else 1 
        second = 0 if folder[:3] == "neg" else 1
        labels.append((first, second))
        text = to_word_list(open("data/" + folder + "/" + f))
        data.append(text)
    return data, labels