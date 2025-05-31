import streamlit as st
import pickle

# Model ve TF-IDF yükle
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

st.title("📝 Consumer Complaint Classifier")
st.subheader("Tahmin Et: Bu şikayet hangi ürünle ilgili?")

# Kullanıcıdan şikayet metni al
complaint_text = st.text_area("Şikayet metnini buraya yazın...")

if st.button("Tahmin Et"):
    if complaint_text.strip() == "":
        st.warning("Lütfen bir metin girin.")
    else:
        # TF-IDF ile dönüştür
        input_vec = vectorizer.transform([complaint_text])
        # Tahmin yap
        prediction = model.predict(input_vec)[0]
        st.success(f"📦 Tahmini Ürün: **{prediction}**")
