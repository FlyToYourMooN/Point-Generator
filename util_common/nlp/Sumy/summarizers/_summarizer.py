# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals


from collections import namedtuple
from operator import attrgetter
from ..utils import ItemsCount
from .._compat import to_unicode
from ..nlp.stemmers import null_stemmer

import numpy as np
from numpy.linalg import norm
import re


SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))


class AbstractSummarizer(object):
    def __init__(self, stemmer=null_stemmer,language="english"):
        if not callable(stemmer):
            raise ValueError("Stemmer has to be a callable object")

        self.language = language.lower()
        if self.language.startswith("en"):
            self._get_sentence_length = get_en_sentence_length
        if self.language.startswith("ch"):
            # calculate the length of the chinese sentence
            # 1 represents the`。`
            self._get_sentence_length = get_cn_sentence_length

        self._stemmer = stemmer

    def __call__(self, document, sentences_count):
        raise NotImplementedError("This method should be overriden in subclass")

    def stem_word(self, word):
        return self._stemmer(self.normalize_word(word))

    def normalize_word(self, word):
        return to_unicode(word).lower()

    def _get_best_sentences(self, sentences, count, rating, *args, **kwargs):
        rate = rating
        if isinstance(rating, dict):
            assert not args and not kwargs
            rate = lambda s: rating[s]

        infos = (SentenceInfo(s, o, rate(s, *args, **kwargs))
            for o, s in enumerate(sentences))

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        if not isinstance(count, ItemsCount):
            count = ItemsCount(count)
        # infos = count(infos)
        # sort sentences by their order in document
        infos = sorted(infos, key=attrgetter("order"))
        # print(tuple([i.sentence,i.rating] for i in infos))
        return tuple([i.sentence,(i.rating,i.order)] for i in infos)#tuple(i.sentence for i in infos)

    @property
    def summary_order(self):
        return self._summary_order

    @summary_order.setter
    def summary_order(self, sum_order):
        self._summary_order = sum_order


    def _cosSim(self, vector1, vector2):
        # both vector are row vectors
        vector1, vector2 = np.mat(vector1), np.mat(vector2)
        numerator = float(vector1 * vector2.T)
        denominator = norm(vector1) * norm(vector2)
        if denominator > 0:
            return numerator / denominator
        else:
            return 0.0


def get_cn_sentence_length(sentence):
    """
    get the actual length of chinese sentence
    :para : Sentence()
    """
    # the length of ', NBA' should be two
    # the length of ',NBA' will be one
    # the same behavior as microsoft word
    chinese_word_pattern = re.compile(u"[\u4e00-\u9fa5。；，：“”（）、？《》]+",
                                    re.UNICODE)
    english_or_number_pattern = re.compile(u"[^\u4e00-\u9fa5\s。；，：“”（）、？《》]+",
                                        re.UNICODE)
    chinese_word_list = re.findall(chinese_word_pattern, sentence._texts)
    english_or_number_list = re.findall(english_or_number_pattern, sentence._texts)
    chinese_len = len(''.join(chinese_word_list))
    english_or_number_len = len(english_or_number_list)
    # 1 represents the '。'
    return chinese_len + english_or_number_len + 1

def get_en_sentence_length(sentence):
    # print(sentence.words)
    words_list = sentence.words
    return len(words_list)