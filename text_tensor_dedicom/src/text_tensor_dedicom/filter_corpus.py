from datetime import datetime
from typing import List

from text_tensor_dedicom.data_classes import Corpus


def filter_dates(corpus: Corpus, start_date: str, end_date: str):
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    for bin_ in corpus:
        filtered_articles = [article for article in bin_ if start_date <= article.date <= end_date]
        bin_.articles = filtered_articles
    return corpus


def filter_sections(corpus: Corpus, filtered_sections: List[str]) -> Corpus:
    for bin_ in corpus:
        filtered_articles = [article for article in bin_ if article.section in filtered_sections]
        bin_.articles = filtered_articles
    return corpus
