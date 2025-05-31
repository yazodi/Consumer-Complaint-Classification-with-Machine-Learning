README.md

# 📝 Consumer Complaint Classification with Machine Learning

Bu proje, tüketicilerin yazdığı şikayet metinlerini analiz ederek, hangi ürünle ilgili olduğunu otomatik olarak tahmin etmeyi amaçlayan bir doğal dil işleme (NLP) projesidir.

Makine öğrenmesi kullanılarak, tüketici şikayetlerinden `Product` sınıfı tahmin edilmiştir (örneğin: Mortgage, Credit Card, Bank Account).

---

## 📌 Proje Hedefi

Kullanıcının yazdığı şikayet metnine göre ilgili ürün kategorisini tahmin etmek.  
Bu sayede müşteri hizmetlerinde sınıflandırma süreci otomatikleştirilebilir.

---

## 🧠 Kullanılan Yöntemler

1. **Veri Okuma ve Temizleme**
   - İlk 100.000 satır kullanıldı (`consumer_complaints.csv`)
   - Eksik (`NaN`) veriler çıkarıldı
   - `Consumer complaint narrative` ve `Product` sütunları kullanıldı

2. **Doğal Dil İşleme (NLP)**
   - TF-IDF vektörleştirme (en fazla 5000 kelime)
   - İngilizce durak kelimeler çıkarıldı

3. **Model Eğitimi**
   - Model: `Logistic Regression`
   - Eğitim/Test oranı: %80 / %20
   - Performans: `classification_report` ile ölçüldü

4. **Model Kaydı**
   - Model + TF-IDF Vectorizer → `model.pkl` dosyasına kaydedildi

5. **Streamlit Uygulaması**
   - Kullanıcıdan metin alır
   - Model tahmini gösterir
   - Anında çalıştırılabilir arayüz

---

## 🚀 Nasıl Çalıştırılır?

### 1. Gereksinimleri yükle
```bash
pip install -r requirements.txt



📦 Kullanılan Dosyalar
Dosya	Açıklama
consumer_complaints_100k.csv	Veri setinin ilk 100.000 satırı
model.py	Model eğitimi ve .pkl kaydı
model.pkl	Eğitilmiş model ve TF-IDF
app.py	Streamlit kullanıcı arayüzü
requirements.txt	Gerekli kütüphaneler listesi
README.md	Proje açıklaması



🧪 Örnek Kullanım

I was charged extra fees and my credit card was closed without notice.
Tahmini ürün:
Credit card or prepaid card


 Eğitim Amacı
Bu proje eğitim amacıyla geliştirilmiştir. Gerçek dünyada kullanılmadan önce daha büyük veri ile model iyileştirme, sınıf dengesizliği çözümü, ileri NLP teknikleri ve hiperparametre optimizasyonu yapılması önerilir

🪪 Lisans
MIT License