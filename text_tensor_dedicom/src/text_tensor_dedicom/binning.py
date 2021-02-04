from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional

from text_tensor_dedicom.data_classes import Article, Bin


def bin_articles(articles: List[Article], bin_by: str = 'month', bin_size: Optional[int] = None) -> List[Bin]:
    if bin_by == 'article':
        return _bin_articles_individually(articles)
    if bin_by == 'quarter':
        return _bin_articles_by_quarter(articles)
    if bin_by == 'month':
        return _bin_articles_by_month(articles)
    if bin_by == 'week':
        return _bin_articles_by_week(articles)
    if bin_by == 'day':
        return _bin_articles_by_day(articles)
    if bin_by == 'even':
        assert bin_size is not None
        return _bin_articles_by_count(articles, bin_size)
    raise NotImplementedError(f'Binning method {bin_by} not implemented')


def _bin_articles_by_day(articles: List[Article]) -> List[Bin]:
    date_to_articles: Dict[datetime.date: List[Article]] = defaultdict(list)
    for article in articles:
        date_to_articles[article.date].append(article)

    bins = []
    for id_, (date, articles) in enumerate(date_to_articles.items()):
        bin_ = Bin(id_=id_,
                   start_date=date,
                   end_date=date,
                   articles=articles)
        bins.append(bin_)
    bins = sorted(bins, key=lambda bin_: bin_.start_date)
    return bins


def _bin_articles_by_week(articles: List[Article]) -> List[Bin]:
    week_to_articles: Dict[int: List[Article]] = defaultdict(list)
    for article in articles:
        week = article.date.isocalendar()[1]
        week_to_articles[week].append(article)

    bins = []
    for id_, (date, articles) in enumerate(week_to_articles.items()):
        bin_ = Bin(id_=id_,
                   start_date=min([article.date for article in articles]),
                   end_date=max([article.date for article in articles]),
                   articles=articles)
        bins.append(bin_)
    bins = sorted(bins, key=lambda bin_: bin_.start_date)
    return bins


def _bin_articles_by_month(articles: List[Article]) -> List[Bin]:
    month_to_articles: Dict[datetime.date: List[Article]] = defaultdict(list)
    for article in articles:
        month = datetime(article.date.year, article.date.month, 1)
        month_to_articles[month].append(article)

    bins = []
    for date, articles in month_to_articles.items():
        month = datetime.strftime(date, '%b %y')
        bin_ = Bin(id_=month,
                   start_date=min([article.date for article in articles]),
                   end_date=max([article.date for article in articles]),
                   articles=articles)
        bins.append(bin_)
    bins = sorted(bins, key=lambda bin_: bin_.start_date)

    return bins


def chunker(seq, size: int):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def _bin_articles_by_quarter(articles: List[Article]) -> List[Bin]:
    month_to_articles: Dict[datetime.date: List[Article]] = defaultdict(list)
    for article in articles:
        month = datetime(article.date.year, article.date.month, 1)
        month_to_articles[month].append(article)

    bins = []
    for batch in chunker(seq=list(month_to_articles.items()), size=3):
        first_months_name = datetime.strftime(batch[0][0], '%b %y')
        last_months_name = datetime.strftime(batch[-1][0], '%b %y')
        quarter_name = f'{first_months_name} - {last_months_name}'
        articles = [article for _, articles in batch for article in articles]

        bin_ = Bin(id_=quarter_name,
                   start_date=min([article.date for article in articles]),
                   end_date=max([article.date for article in articles]),
                   articles=articles)
        bins.append(bin_)
    bins = sorted(bins, key=lambda bin_: bin_.start_date)

    return bins


def _bin_articles_individually(articles: List[Article]) -> List[Bin]:
    articles = sorted(articles, key=lambda x: x.date)
    bins = []
    for id_, article in enumerate(articles):
        bin_ = Bin(id_=id_,
                   start_date=article.date,
                   end_date=article.date,
                   articles=[article])
        bins.append(bin_)
    bins = sorted(bins, key=lambda bin_: bin_.start_date)
    return bins


def _bin_articles_by_count(articles: List[Article], bin_size) -> List[Bin]:
    articles = sorted(articles, key=lambda x: x.date)
    bins = []
    for id_, i in enumerate(range(0, len(articles), bin_size)):
        articles_bin = articles[i:i + bin_size]
        bin_ = Bin(id_=id_,
                   start_date=min([article.date for article in articles_bin]),
                   end_date=max([article.date for article in articles_bin]),
                   articles=articles_bin)
        bins.append(bin_)
    bins = sorted(bins, key=lambda bin_: bin_.start_date)
    return bins
