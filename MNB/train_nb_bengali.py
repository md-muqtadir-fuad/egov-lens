# -*- coding: utf-8 -*-
"""
Bengali Sentiment Analysis (Multinomial Naive Bayes)
- Cleans + tokenizes Bangla text (with optional stemming if bangla-stemmer is installed)
- TF-IDF features (word + character n-grams)
- Stratified train/val/test split
- Grid search over alpha and n-gram ranges
- Final evaluation + model export
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from typing import List, Iterable, Set

STOPWORDS: Set[str] = set()

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from joblib import dump

# Try optional Bangla stemmer
try:
    from bangla_stemmer.stemmer import BanglaStemmer
    _BN_STEMMER = BanglaStemmer()
except Exception:
    _BN_STEMMER = None


# -----------------------------
# Text utilities for Bangla
# -----------------------------
BN_DIGITS = "০১২৩৪৫৬৭৮৯"
EN_DIGITS = "0123456789"

BN_PUNCT = "।॥‘’“”—–…•"
EXTRA_PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
PUNCT_TABLE = str.maketrans({c: " " for c in BN_PUNCT + EXTRA_PUNCT})

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.UNICODE)
MENTION_RE = re.compile(r"[@#][\w\-_]+", re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+", re.UNICODE)

def bn_normalize(text: str) -> str:
    """Basic normalization for Bangla social text."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    # normalize punctuation -> space
    t = t.translate(PUNCT_TABLE)
    # normalize digits to one script (choose English)
    for bd, ed in zip(BN_DIGITS, EN_DIGITS):
        t = t.replace(bd, ed)
    # collapse spaces
    t = MULTISPACE_RE.sub(" ", t)
    return t.strip()


def bn_tokenize(text: str, remove_stopwords: bool = True, stem: bool = False) -> List[str]:
    """Whitespace tokenization after normalization, with optional stopword removal and stemming."""
    t = bn_normalize(text)
    tokens = [tok for tok in t.split(" ") if tok]
    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
    if stem and _BN_STEMMER is not None:
        tokens = [_BN_STEMMER.stem(w) for w in tokens]
    return tokens


# -----------------------------
# Sklearn transformer wrapper
# -----------------------------
class BengaliCleaner(BaseEstimator, TransformerMixin):
    """Applies bn_normalize; returns string so Vectorizer can tokenize or we pass our tokenizer."""

    def __init__(self, remove_stopwords: bool = True, stem: bool = False):
        self.remove_stopwords = remove_stopwords
        self.stem = stem

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        # Return cleaned strings for vectorizers that use our tokenizer
        return [bn_normalize(x) for x in X]

# -----------------------------
# Stopwords loader
# -----------------------------
def load_stopwords(path: str) -> Set[str]:
    if not path or not os.path.isfile(path):
        print("[info] No external stopwords file provided; proceeding without stopwords filtering.")
        return set()
    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.add(line)
    print(f"[info] Loaded {len(words)} stopwords from {path}")
    return words

# -----------------------------
# Data loader (with sample fallback)
# -----------------------------
SAMPLE_ROWS = [
    ["কীটনাশকের গায়ে ১০০ লেখা থাকলে ৪০ টাকায় পাওয়া যায়", "neutral"],
    ["এলপিজির গায়ে মূল্য কেন দেওয়া হয় না, আজও বুঝতে পারি না", "negative"],
    ["সারের বস্তার গায়ে মূল্য লেখাটাও বাধ্যতামূলক করুন", "positive"],
    ["গায়ে লেখা দামে বেশির ভাগ পণ্য বিক্রি হয় না, বেশি দামে কিনতে হয়, এটা দেখার লোক নেই", "negative"],
    ["ঔষধের প্রত্যেকটি পাতায় মূল্য লেখা থাকলে প্রতারণা থেকে রক্ষা পাওয়া যাবে", "positive"],
    ["ভারতীয় পণ্য বর্জন করুন এবং পুরোপুরি কার্যকর করুন", "positive"],
    ["দেশের মানুষ কোটি কোটি টাকা চায় না, তারা চায় পেট ভরে দুবার খেতে, খাদ্যের দাম কমে গরীব ও মধ্যবিত্ত মানুষের দিকে তাকিয়ে", "positive"],
    ["সঠিক পদক্ষেপ নেওয়া হয়েছে", "positive"],
    ["অ-কর্মা মন্ত্রী, কাজ হবে না", "negative"],
    ["খালি বলে যাচ্ছে, বাজার নিয়ন্ত্রণ হচ্ছে না, গরিব মানুষ বাঁচতে চায়", "negative"],
    ["এই মেশিন কি ভাবে নেওয়া যাবে, সফিউল ভাইয়ের সাথে যোগাযোগ করার কোনো উপায় আছে কি?", "neutral"],
    ["সাত মাছ চাষের এই ডিভাইসটি কি বাংলাদেশে পাওয়া যাবে", "neutral"],
    ["এই ডিভাইসটি কি এখনো মার্কেটে আসে নাই, কোথায় পাওয়া যাবে জানালে উপকৃত হবো", "neutral"],
    ["কবে নাগাদ এই স্মার্ট টেকনোলজি পাব, আপনাকে অনেক ধন্যবাদ", "neutral"],
    ["এই ডিভাইস এখন কি মার্কেটে আছে, থাকলে কেউ জানান", "neutral"],
]
SAMPLE_DF = pd.DataFrame(SAMPLE_ROWS, columns=["text", "sentiment"])

def load_dataset(path: str) -> pd.DataFrame:
    if os.path.isfile(path):
        df = pd.read_csv(path)
        # Expect at least 'text' and 'sentiment'
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        if "text" not in df.columns or "sentiment" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'sentiment' columns.")
        # Drop obvious empties
        df = df.dropna(subset=["text", "sentiment"])
        # Trim
        df["text"] = df["text"].astype(str).str.strip()
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
        # Keep only desired classes (positive/negative/neutral)
        df = df[df["sentiment"].isin(["positive", "negative", "neutral"])].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No rows left after filtering valid sentiments.")
        return df
    else:
        print(f"[warn] File '{path}' not found. Falling back to small in-script sample.")
        return SAMPLE_DF.copy()


# -----------------------------
# Build model
# -----------------------------
def build_pipeline(use_stem: bool = False) -> Pipeline:
    """
    Mixed features:
      - word-level TF-IDF using our custom tokenizer (good for semantics)
      - char-level TF-IDF 3-5 grams (robust to spelling/inflection)
    """
    word_vec = TfidfVectorizer(
        tokenizer=lambda s: bn_tokenize(s, remove_stopwords=True, stem=use_stem),
        preprocessor=lambda s: s,
        token_pattern=None,          # <-- add this when using a custom tokenizer
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2
    )

    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3,5),
        min_df=2
    )

    features = FeatureUnion([
        ("word", word_vec),
        ("char", char_vec),
    ])

    pipe = Pipeline([
        ("clean", BengaliCleaner()),
        ("features", features),
        ("clf", MultinomialNB())
    ])
    return pipe


def train_and_eval(df: pd.DataFrame, out_dir: str, use_stem: bool):
    X = df["text"].tolist()
    y = df["sentiment"].tolist()

    # Split: 70/15/15 with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    pipe = build_pipeline(use_stem=use_stem)

    # Reasonable grid for NB + n-grams (keep small for speed)
    param_grid = {
        "features__word__ngram_range": [ (1,1), (1,2) ],
        "features__word__min_df": [1, 2],
        "features__char__ngram_range": [ (3,5), (3,6) ],
        "clf__alpha": [0.2, 0.5, 1.0, 2.0],
    }

    min_class = pd.Series(y_train).value_counts().min()
    cv = StratifiedKFold(n_splits=max(2, min(5, int(min_class))), shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )
    gs.fit(X_train, y_train)

    print("\nBest params:", gs.best_params_)
    print("Best CV f1_macro: %.4f" % gs.best_score_)

    # Validation set performance
    yv_pred = gs.predict(X_val)
    print("\n=== Validation Metrics ===")
    print("Accuracy: %.4f" % accuracy_score(y_val, yv_pred))
    print("F1 Macro: %.4f" % f1_score(y_val, yv_pred, average="macro"))
    print(classification_report(y_val, yv_pred, digits=4))

    # Final test evaluation
    yt_pred = gs.predict(X_test)
    print("\n=== Test Metrics ===")
    print("Accuracy: %.4f" % accuracy_score(y_test, yt_pred))
    print("F1 Macro: %.4f" % f1_score(y_test, yt_pred, average="macro"))
    print(classification_report(y_test, yt_pred, digits=4))
    print("Confusion matrix (test):\n", confusion_matrix(y_test, yt_pred, labels=["negative","neutral","positive"]))

    # Export model
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "nb_bengali_sentiment.joblib")
    dump(gs.best_estimator_, model_path)
    print(f"\nSaved model to: {model_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset.csv",
                    help="CSV with 'text' and 'sentiment' columns.")
    ap.add_argument("--out", type=str, default="artifacts",
                    help="Where to save the model.")
    ap.add_argument("--stem", action="store_true",
                    help="Turn on Bangla stemming (requires bangla-stemmer).")
    ap.add_argument("--stopwords", type=str, default="",
                    help="Path to UTF-8 stopwords file (one word per line).")
    args = ap.parse_args()

    # load external stopwords into the global used by bn_tokenize()
    global STOPWORDS
    STOPWORDS = load_stopwords(args.stopwords)

    df = load_dataset(args.data)
    print(df["sentiment"].value_counts())
    train_and_eval(df, out_dir=args.out, use_stem=args.stem)

if __name__ == "__main__":
    main()
# End of file