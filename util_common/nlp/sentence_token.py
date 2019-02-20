
from .Sumy.summarizers.text_rank import TextRankSummarizer
from .Sumy.nlp.stemmers import Stemmer
from .Sumy.utils import get_stop_words
from .Sumy.parsers.html import HtmlParser
from .Sumy.parsers.plaintext import PlaintextParser
from .Sumy.nlp.tokenizers import Tokenizer


def sentenceTaken(language, text):
    sentences = []
    summarizer = TextRankSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    for item in parser.document.sentences:
        sentences.append("{}".format(item))
    # print(sentences)
    return sentences


if __name__ == "__main__":
    print(sentenceTaken("chinese",text))