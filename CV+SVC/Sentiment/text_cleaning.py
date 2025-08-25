import re
import string
from pathlib import Path
from typing import List, Set, Iterable
from sklearn.base import BaseEstimator, TransformerMixin

# Bengali digits
BN_DIGITS = '০১২৩৪৫৬৭৮৯'
EN_DIGITS = '0123456789'

# Bengali + extra punctuations
BN_PUNCT = '।॥‘’“”—–…•'
EXTRA_PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
PUNCT_TABLE = str.maketrans({c: ' ' for c in BN_PUNCT + EXTRA_PUNCT})

# Regex patterns
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.UNICODE)
MENTION_RE = re.compile(r'[@#][\w\-_]+', re.UNICODE)
MULTISPACE_RE = re.compile(r'\s+', re.UNICODE)

# Stopwords (optional, load externally)
STOPWORDS = set()
_BN_STEMMER = None   # Placeholder if you want to use a Bengali stemmer


def bn_normalize(text: str) -> str:
    """Normalize Bangla text by removing URLs, mentions, punctuation, digits, extra spaces."""
    if not isinstance(text, str):
        return ''
    t = text.strip()
    t = URL_RE.sub(' ', t)
    t = MENTION_RE.sub(' ', t)
    t = t.translate(PUNCT_TABLE)
    for bd, ed in zip(BN_DIGITS, EN_DIGITS):
        t = t.replace(bd, ed)
    t = MULTISPACE_RE.sub(' ', t)
    return t.strip()


def load_stopwords(path: Path) -> Set[str]:
    """Load stopwords from external file."""
    if not path or not path.exists():
        print('[info] No external stopwords file found; proceeding without stopwords filtering.')
        return set()
    words = set()
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            words.add(line)
    print(f'[info] Loaded {len(words)} stopwords from {path}')
    return words


def bn_tokenize(text: str, remove_stopwords: bool = True, stem: bool = False) -> List[str]:
    """Tokenize normalized Bangla text."""
    t = bn_normalize(text)
    tokens = [tok for tok in t.split(' ') if tok]
    if remove_stopwords and STOPWORDS:
        tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
    if stem and _BN_STEMMER is not None:
        tokens = [_BN_STEMMER.stem(w) for w in tokens]
    return tokens


class BengaliCleaner(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer for Bangla text cleaning (can be used in Pipelines)."""
    def __init__(self): 
        pass
    
    def fit(self, X: Iterable[str], y=None): 
        return self
    
    def transform(self, X: Iterable[str]) -> List[str]:
        return [bn_normalize(x) for x in X]
