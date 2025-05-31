












import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# 1. Veri yükleme
df = pd.read_csv("consumer_complaints_100k.csv")

# 2. Sütun adlarını düzelt
df.columns = df.columns.str.strip()

# 3. Gerekli sütunları seç (Boş olmayanlar)
df = df[['Consumer complaint narrative', 'Product']].dropna()

# 4. Giriş ve hedef verileri ayır
X = df['Consumer complaint narrative']
y = df['Product']

# 5. TF-IDF vektörleştirici
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# 6. Eğitim / test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 7. Model oluştur ve eğit
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Performans raporu
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 9. Model ve vectorizer'ı kaydet
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("\n✅ Model başarıyla eğitildi ve 'model.pkl' dosyasına kaydedildi.")












# İlk 100.000 satırı kullanır

# Sütun isimlerini düzeltir

# Eksik verileri temizler

# TF-IDF ile metni sayısallaştırır

# Logistic Regression ile model eğitir

# Modeli ve TF-IDF vektörizerini .pkl dosyasına kaydeder