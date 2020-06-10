import string
from typing import Optional, List

from nltk import SnowballStemmer, word_tokenize


class Preprocessor:
    def __init__(self,
                 pipeline: Optional[List] = None):
        self.pipeline = pipeline

    @staticmethod
    def to_lower(text: str) -> str:
        return text.lower()

    @staticmethod
    def remove_punctuation(text: str) -> str:
        punct_removal = string.punctuation + '’“'
        return text.translate(str.maketrans('', '', punct_removal))

    @staticmethod
    def remove_digits(text: str) -> str:
        return text.translate(str.maketrans('', '', string.digits))

    @staticmethod
    def stem(text: str) -> str:
        stemmer = SnowballStemmer(language='german')
        return ' '.join([stemmer.stem(word) for word in text.split()])

    @staticmethod
    def remove_multi_spaces(text: str) -> str:
        return ' '.join(text.split())

    @staticmethod
    def add_whitespace_after_period(text: str) -> str:
        return text.replace('.', '. ')

    @staticmethod
    def remove_single_chars(text: str) -> str:
        return ' '.join([word for word in text.split() if len(word) > 1])

    @staticmethod
    def remove_stopwords(text: str) -> str:
        sw = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren",
              "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
              "can",
              "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don",
              "don't",
              "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn",
              "hasn't",
              "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
              "how",
              "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma",
              "me",
              "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor",
              "not", "nt",
              "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
              "own",
              "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so",
              "some",
              "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there",
              "these",
              "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn",
              "wasn't",
              "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
              "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your",
              "yours",
              "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm",
              "i've",
              "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've",
              "we'd",
              "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would",
              "'s", "'ve", "'ll", "'d", "'m", "'t", "'re"]
        return ' '.join([token for token in word_tokenize(text) if token.lower() not in sw])

    def process(self, text: str) -> str:
        if isinstance(self.pipeline, List):
            for step in self.pipeline:
                text = getattr(Preprocessor, step)(text)
        else:
            print('No Preprocessing is done.')
        return text
