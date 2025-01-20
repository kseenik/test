#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pytest
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


nltk.download("stopwords")
original_data = pd.read_csv("balanced_spam.csv")
attacked_data = pd.read_csv("attacked.csv")


# In[4]:


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


# In[5]:


# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[6]:


# Предсказания модели
y_pred = model.predict(X_test)
y_attacked_pred = model.predict(X_attacked)


# In[7]:


# Тестирование предобработки данных
def test_preprocess_text():
    raw_text = "Check out the website https://example.com! This is the first text with number 123."
    expected_result = "check websit first text number"
    assert preprocess_text(raw_text) == expected_result

# Тестирование модели
def test_model_prediction():
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.8  # Ожидаемая точность должна быть выше 80%

# Тестирование метрик качества
def test_classification_report():
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']
    assert accuracy > 0.8  # Проверка точности

def test_confusion_matrix():
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_attacked, y_attacked_pred)
    assert conf_matrix.shape == (2, 2)  # Матрица ошибок должна быть 2x2

# Визуализация матрицы ошибок (для наглядности)
def plot_confusion_matrix():
    conf_matrix = confusion_matrix(y_attacked, y_attacked_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
    plt.title('Confusion Matrix (Attacked Dataset)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == "__main__":
    pytest.main()

