
from util_common.nlp.word_dictionary import word_dictionary
from util_common.nlp.os import read_folder_content, read_content
from util_common.mongo.nlp import nlp_chinese_inter_content

from lib.dataset.generator import ChineseDataset

import torch

def read_configure_word_index(path):
    # Declare the hyperparameter
    configure = eval(read_content(path))

    # Get the word vs index
    word_index, index_word = word_dictionary(sentences=nlp_chinese_inter_content(), 
                                             language=configure["language"])
    return configure, word_index, index_word


def dataset_pipline(folder, batch_size, word_index, configure, shuffle, train):
    # Declare the dataset pipline
    dataset = ChineseDataset(folder, configure, word_index, train)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle)
    return loader


def processing(path):
    # 1. Declare the device
    device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")

    # 2. Declare the processing
    configure, word_index, index_word = read_configure_word_index(path)
    
    # 3. Declare the dataset pipline
    train_loader = dataset_pipline(folder=configure["folder"], 
                                   batch_size=configure["batch_size"],
                                   word_index=word_index, 
                                   configure=configure, shuffle=True, train=True)

    test_loader = dataset_pipline(folder=configure["folder"],
                                  batch_size=configure["batch_size"],
                                  word_index=word_index,
                                  configure=configure, shuffle=True, train=True)

    return device, configure, word_index, index_word,train_loader, test_loader
    
