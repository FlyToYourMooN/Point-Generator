
import torch.utils.data as data 
import torch

from util_common.nlp.os import read_content
from util_common.nlp.word_token import word_token
from util_common.nlp.word_dictionary import word2index

import os
import numpy as np



class KaggleDataset(data.Dataset):

    def __init__(self, folder, configure, word_index):
        # Declare the hyperparameter
        self.folder = folder
        self.language = configure["language"]
        self.word_index = word_index
        self.max_content = configure["max_content"]
        self.max_target = configure["max_output"]
        self.pad_index = word_index["<pad>"]

        # Files collection
        self.files = os.listdir(folder)


    def __getitem__(self, index):

        # 1.Picked one file
        file = self.files[index]

        # 2.Read the content of file
        input_txt, target_txt = read_content(self.folder+file).split("<label:>")
        
        # 3.Tranfer the words to index
        transfer_input, tranfer_target = list(map(lambda x: self.transfer_content(x), 
                                                  [input_txt, target_txt]))

        # 4.Padding the data
        transfer_input = self.padding(transfer_input, data_type="content")
        tranfer_target = self.padding(tranfer_target, data_type="target")

        return (transfer_input, tranfer_target)


    def padding(self, data, data_type="content"):
        # padding to max_content
        if data_type == "content":
            max_length = self.max_content
        else: 
            max_length = self.max_target
        len_data = min([max_length,len(data)])
        data_pad = np.pad(data, (0,max(0,max_length-len_data)), 'constant', constant_values=(self.pad_index))[:max_length]
        return data_pad


    def transfer_content(self, content):
        """ Transfer words to index
        # Arguments
            - content: [w1, w2,...]
        
        # Returns
            - transfer_words: [1,2,...]
        """
        words = word_token(text=content, language=self.language)
        words += ["<eos>"] 
        transfer_words = word2index(words, self.word_index)
        return transfer_words

    def __len__(self):
        return len(self.files)



from util_common.mongo.nlp import nlp_chinese_inter

class ChineseDataset(data.Dataset):
    
    def __init__(self, folder, configure, word_index, train):
        # Declare the hyperparameter
        if train:
            self.ids, self.collection = nlp_chinese_inter()
            self.ids = self.ids[:-25]
        else: 
            self.ids, self.collection = nlp_chinese_inter()
            self.ids = self.ids[-25:]
        self.language = configure["language"]
        self.word_index = word_index
        self.max_content = configure["max_content"]
        self.max_target = configure["max_output"]
        self.pad_index = word_index["<pad>"]


    def __getitem__(self, index):

        # 1.Picked one file
        id = self.ids[index]
        data = self.collection.find_one({"_id":id})

        # 2.Read the content of file
        input_txt, target_txt = (data["input"], data["target"])
        
        # 3.Tranfer the words to index
        transfer_input, tranfer_target = list(map(lambda x: self.transfer_content(x), 
                                                  [input_txt, target_txt]))

        # 4.Padding the data
        transfer_input = self.padding(transfer_input, data_type="content")
        tranfer_target = self.padding(tranfer_target, data_type="target")

        return (transfer_input, tranfer_target)


    def padding(self, data, data_type="content"):
        # padding to max_content
        if data_type == "content":
            max_length = self.max_content
        else: 
            max_length = self.max_target
        len_data = min([max_length,len(data)])
        data_pad = np.pad(data, (0,max(0,max_length-len_data)), 'constant', constant_values=(self.pad_index))[:max_length]
        return data_pad


    def transfer_content(self, content):
        """ Transfer words to index
        # Arguments
            - content: [w1, w2,...]
        
        # Returns
            - transfer_words: [1,2,...]
        """
        words = word_token(text=content, language=self.language)
        words += ["<eos>"] 
        transfer_words = word2index(words, self.word_index)
        return transfer_words

    def __len__(self):
        return len(self.ids)
