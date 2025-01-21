# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12mYp2BVp4c-h1YiqoJbF10TUi1m1N5ot
"""

import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords")
original_data = pd.read_csv("balanced_spam.csv")
attacked_data = pd.read_csv("attacked.csv")

# Валидация данных
def validate_input_data(data):
    if data is None or len(data) == 0:
        raise ValueError("Input data is empty or None.")
    if not isinstance(data, pd.Series):
        raise TypeError("Input data should be of type pandas.Series.")
    if any(data.str.len() < 1):
        raise ValueError("Some input texts are too short.")
    return data


# Функция предобработки текста
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)


# Применяем предобработку данных
original_data["text"] = original_data["text"].apply(preprocess_text)
attacked_data["text"] = attacked_data["text"].apply(preprocess_text)

# Векторизация текста
vectorizer = TfidfVectorizer(max_features=5000)
X_original = vectorizer.fit_transform(original_data["text"]).toarray()
y_original = original_data["spam"]
X_attacked = vectorizer.transform(attacked_data["text"]).toarray()
y_attacked = attacked_data["spam"]

# Балансировка классов
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_original, y_original)

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Предсказания модели
y_pred = model.predict(X_test)
y_attacked_pred = model.predict(X_attacked)

