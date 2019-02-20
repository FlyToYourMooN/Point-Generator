

from util_docomo_beijing.nlp.os import read_folder_content
from util_docomo_beijing.nlp.word_token import word_token

class word_dictionary_cell():

    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.index = 0
        self.insert_go_eos()
    
    def insert(self, word):
        if word not in self.word_index:
            self.word_index[word] = self.index 
            self.index_word[self.index] = word
            self.index += 1
    
    def insert_go_eos(self):
        words = ["<go>","<eos>","<unknown>","<pad>"]
        for word in words:
            self.insert(word)


def word_dictionary(sentences, language):
    """
    # Arguments
        sentences {list}: [s1,s2,s3]
    
    # Returns
        word_index {dict}
        index_word {dict}

    """
    dictionary = word_dictionary_cell()

    for sentence in sentences:
        words = word_token(text=sentence, language=language)
        for word in words:
            dictionary.insert(word)
    
    word_index = dictionary.word_index
    index_word = dictionary.index_word
    return word_index, index_word


def word2index(words, word_index):
    """
    # Arguments
        words {list}: [w1,w2]
        word_index {dict}: {"w1":1}
    
    # Returns
        [1,2,3,4]

    """
    result = []
    for word in words:
        if word in word_index:
            result.append(word_index[word])
        else: 
            result.append(word_index["<null>"])
    return result


if __name__ == "__main__":
    path = "../dataset/tv20tp/train/"
    language = "japanese"
    sentences = read_folder_content(path)
    word_index, index_word = word_dictionary(sentences, language)
    print(word_index)