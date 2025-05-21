# Trendyol Yorumları ile Duygu Analizi Projesi

# Gerekli kütüphanelerin kurulumu
# pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Veri Seti Hazırlığı ---

# Olumlu ve olumsuz yorumlar
positive_comments = [
    "Çok kaliteli ürün, herkese tavsiye ederim.",
    "Kargom çok hızlı geldi, paketleme mükemmeldi.",
    "Beklentimin çok üstündeydi.",
    "Görseldeki ile birebir aynı geldi.",
    "Ürün anlatıldığı gibi ve oldukça iyi.",
    "Kesinlikle tekrar alırım.",
    "Hediye olarak aldım, çok beğenildi.",
    "Fiyat performans açısından harika.",
    "Tertemiz ve düzgün paketlenmişti.",
    "Memnun kaldım, teşekkür ederim."
]

negative_comments = [
    "Çok kötü çıktı, param boşa gitti.",
    "Ürün bozuk geldi, iade ettim.",
    "Beklediğim gibi değildi.",
    "Görsel ile alakasi yok.",
    "Kalitesiz malzeme, tavsiye etmiyorum.",
    "Hatalı ürün göndermişler.",
    "Kırık geldi, şikayetçiyim.",
    "Kargo çok gecikti.",
    "Paketleme özensizdi.",
    "Kesinlikle tavsiye etmiyorum."
]

# 100 olumlu ve 100 olumsuz yorum
sample_data = {
    "yorum": random.choices(positive_comments, k=100) + random.choices(negative_comments, k=100),
    "label": [1]*100 + [0]*100
}

df = pd.DataFrame(sample_data)

# --- Veri Seti Bölme ---
X = df["yorum"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Modelleme ---

# Naive Bayes pipeline
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# SVM pipeline
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

# Modelleri eğit
nb_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)

# Tahminler
nb_preds = nb_pipeline.predict(X_test)
svm_preds = svm_pipeline.predict(X_test)

# --- Değerlendirme ---

# Naive Bayes raporu
print("Naive Bayes Classification Report")
print(classification_report(y_test, nb_preds))

# SVM raporu
print("\nSVM Classification Report")
print(classification_report(y_test, svm_preds))

# Confusion Matrix
nb_cm = confusion_matrix(y_test, nb_preds)
svm_cm = confusion_matrix(y_test, svm_preds)

# Confusion Matrix grafiklerle gösterim
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(nb_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Naive Bayes - Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("SVM - Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.show()
