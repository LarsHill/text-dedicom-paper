from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import List, Dict, Optional, Union


@dataclass
class Blob:
    id_: int
    tag: str
    value: str
    value_processed: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict, id_: int = None):
        if 'id_' not in d.keys():
            d['id_'] = id_
        if 'text' in d.keys():
            d['value'] = d.pop('text')
        return cls(**d)

    def to_dict(self) -> Dict:
        return {'id_': self.id_,
                'tag': self.tag,
                'value': self.value}


@dataclass
class Article:
    id_: Optional[int] = None
    title: Optional[str] = None
    url: Optional[str] = None
    author: Optional[str] = None
    section: Optional[str] = None
    _date: Optional[datetime.date] = None
    time_published: Optional[datetime] = None
    time_updated: Optional[datetime] = None
    abstract: Optional[str] = None
    value: Optional[str] = None
    value_processed: Optional[str] = None
    blobs: List[Blob] = field(default_factory=list)
    rating: Optional[float] = None
    split: Optional[str] = None

    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, idx: int) -> Blob:
        return self.blobs[idx]

    @property
    def date(self):
        if self._date is not None:
            return self._date
        return datetime.date(self.time_published) if self.time_published is not None else datetime.date(datetime.strptime('19000101', '%Y%m%d'))

    @classmethod
    def from_dict(cls, d: Dict, id_: int = None):
        if 'value' not in d.keys():
            d['value'] = ' '.join([blob['value'] if 'value' in blob.keys() else blob['text'] for blob in d['blobs']])
        if 'id_' not in d.keys():
            d['id_'] = id_
        if 'time_published' in d.keys() and d['time_published']:
            d['time_published'] = datetime.strptime(d['time_published'][:19], '%Y-%m-%dT%H:%M:%S')
        if 'time_updated' in d.keys() and d['time_updated']:
            d['time_updated'] = datetime.strptime(d['time_updated'][:19], '%Y-%m-%dT%H:%M:%S')
        if '_date' in d:
            d['_date'] = cls.date_from_string(d['_date'])
        elif 'date' in d:
            d['_date'] = cls.date_from_string(d['date'])
            del d['date']
        d['blobs'] = [Blob.from_dict(d=blob, id_=i) for i, blob in enumerate(d['blobs'])]
        return cls(**d)

    def to_dict(self) -> Dict:
        time_published = datetime.strftime(self.time_published,
                                           '%Y-%m-%dT%H:%M:%S') if self.time_published else None
        time_updated = datetime.strftime(self.time_updated,
                                         '%Y-%m-%dT%H:%M:%S') if self.time_updated else None

        return {'id_': self.id_,
                'title': self.title,
                'url': self.url,
                'author': self.author,
                'section': self.section,
                'time_published': time_published,
                'time_updated': time_updated,
                'abstract': self.abstract,
                'value': self.value,
                'value_processed': self.value_processed,
                'blobs': [blob.to_dict() for blob in self.blobs],
                'rating': self.rating,
                'split': self.split}

    @staticmethod
    def date_from_string(date: str) -> Union[datetime.date, str]:
        try:
            return datetime.strptime(date, '%B %d, %Y').date()
        except ValueError:
            try:
                return datetime.strptime('2020 ' + date, '%Y %B %d').date()
            except ValueError:
                try:
                    return datetime.strptime('2020 ' + date, '%Y %b. %d').date()
                except ValueError:
                    return date

    def __repr__(self):
        return f'Article: {self.title[:100]} ...'


@dataclass
class Bin:
    id_: str
    start_date: Optional[Union[datetime, str]] = None
    end_date: Optional[Union[datetime, str]] = None
    articles: List[Article] = field(default_factory=list)

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx: int) -> Article:
        return self.articles[idx]

    @classmethod
    def from_dict(cls, d: Dict, id_: str = None):
        if 'id_' not in d.keys():
            d['id_'] = id_
        d['start_date'] = datetime.strptime(d.get('start_date'), '%Y%m%d') if d.get('start_date') is not None else None
        d['end_date'] = datetime.strptime(d.get('end_date'), '%Y%m%d') if d.get('end_date') is not None else None
        d['articles'] = [Article.from_dict(d=article, id_=i) for i, article in enumerate(d['articles'])]
        return cls(**d)

    def to_dict(self) -> Dict:
        return {'id_': self.id_,
                'start_date': datetime.strftime(self.start_date, '%Y%m%d') if self.start_date else None,
                'end_date': datetime.strftime(self.end_date, '%Y%m%d') if self.end_date else None,
                'articles': [article.to_dict() for article in self.articles]}

    def __repr__(self):
        return f'Bin with {len(self.articles)} articles from {self.start_date} to {self.end_date}.'


@dataclass
class Corpus:
    id_: str
    query: Optional[str] = None
    types: Optional[str] = None
    sections: Optional[str] = None
    sort: Optional[str] = None
    bins: List[Bin] = field(default_factory=list)
    bin_names: Optional[List[str]] = None

    def __len__(self):
        return len(self.bins)

    def __getitem__(self, idx: int) -> Bin:
        return self.bins[idx]

    def __repr__(self):
        return f'Corpus with {len(self.articles)} articles in {len(self.bins)} bins.'

    @classmethod
    def from_dict(cls, d: Dict):
        d['bins'] = [Bin.from_dict(d=bin_, id_=str(i)) for i, bin_ in enumerate(d['bins'])]
        return cls(**d)

    def to_dict(self) -> Dict:
        data = {'id_': self.id_,
                'bins': [bin_.to_dict() for bin_ in self.bins]}
        if self.bin_names is not None:
            data['bin_names'] = self.bin_names
        return data

    @classmethod
    def from_json(cls, path: str):
        return cls.from_dict(json.load(open(path, 'r')))

    def to_json(self, path: str):
        json.dump(self.to_dict(), open(path, 'w'))

    @property
    def articles(self):
        return [article for bin_ in self.bins for article in bin_.articles]
