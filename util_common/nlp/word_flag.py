import jieba.posseg


def word_flag(text, language):
    """ Read a text and return the word flag
    
    Arguments:
        text {[str]} -- [description]
        language {[str]} -- [description]
    
    Returns:
        [list] -- [description]
    """

    result = []
    for item in jieba.posseg.cut(text.strip()):
        result.append((item.word, item.flag))
    return result
