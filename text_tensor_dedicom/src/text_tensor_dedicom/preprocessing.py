import string
from typing import Optional, List

from tqdm import tqdm
import nltk
from nltk import SnowballStemmer, word_tokenize

from text_tensor_dedicom.data_classes import Corpus, Article, Bin


class Preprocessor:
    stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are",
                  "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
                  "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn",
                  "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
                  "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here",
                  "hers", "herself", "him",
                  "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself",
                  "just", "ll", "m",
                  "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn",
                  "needn't", "no",
                  "nor", "not", "nt", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
                  "ourselves", "out",
                  "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn",
                  "shouldn't",
                  "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves",
                  "then",
                  "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve",
                  "very", "was",
                  "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who",
                  "whom", "why",
                  "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're",
                  "you've",
                  "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd",
                  "i'll",
                  "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll",
                  "they're",
                  "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's",
                  "would",
                  "'s", "'ve", "'ll", "'d", "'m", "'t", "'re"]

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
        sw = Preprocessor.stop_words
        try:
            text = ' '.join([token for token in word_tokenize(text) if token.lower() not in sw])
        except LookupError:
            nltk.download('punkt')
            text = ' '.join([token for token in word_tokenize(text) if token.lower() not in sw])
        return text

    @staticmethod
    def remove_empty_reviews(bin_: Bin) -> List[Article]:
        return [review for review in bin_ if not review.value_processed == '']

    def process_blob(self, blob: str) -> str:
        if isinstance(self.pipeline, List):
            for step in self.pipeline:
                blob = getattr(Preprocessor, step)(blob)
        else:
            print('No Preprocessing is done.')
        return blob

    def process_article(self, article: Article):
        article.value_processed = self.process_blob(article.value)

    def process(self,
                corpus: Corpus) -> Corpus:
        for bin_ in tqdm(corpus):
            for article in bin_:
                self.process_article(article=article)
            bin_.articles = Preprocessor.remove_empty_reviews(bin_=bin_)

        return corpus
