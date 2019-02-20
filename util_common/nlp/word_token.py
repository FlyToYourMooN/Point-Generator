
from .pos import pos_japanese

import re
import warnings
import jieba

from nltk.tokenize import RegexpTokenizer

def word_token(text, language):
    """Word token
    
    Arguments:
        text {[str]} -- [description]
        langurage {[str]} -- [description]
    
    Returns:
        [list] -- [description]
    """
    if language == "chinese":
        # r = r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+"
        r = r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！;:？、~@#￥%……&*（）]+"
        line = re.sub(r, '', text)
        list_words = list(jieba.cut(line))
        return list_words
    elif language == "english":
        tokenizer = RegexpTokenizer(r'\w+')
        list_words = tokenizer.tokenize(text)
        return list_words
        
    elif language == "japanese":
        ls_word = []
        ls_pos = []
        data = pos_japanese(text)
        for item in data:
            if "記号" not in item[1]:
                ls_word.append(item[0])
                ls_pos.append(item[1])
        return ls_word
    elif language == "jp_pos":
        ls_word = []
        ls_pos = []
        data = pos_japanese(text)
        for item in data:
            ls_word.append(item[0])
            ls_pos.append(item[1])
        return ls_pos
    elif language == "jp_pos_word":
        words = []
        data = pos_japanese(text)
        for item_pair in data:
            if "名詞" in item_pair[1] or "動詞" in item_pair[1]:
                words.append(item_pair[1])
            elif "記号" in item_pair[1]:
                pass
            else: 
                words.append(item_pair[0])
        return words
