
import streamlit as st
import requests
import pandas as pd

# Konfigurasi halaman
st.set_page_config(
    page_title="Machine Learning Prediction",
    page_icon="🤖",
    layout="centered"
)

# URL Flask API di Vercel
API_URL = "https://flask-model-ten.vercel.app/api/predict"

st.title("🤖 Machine Learning Prediction")
st.write(
    "Masukkan data untuk mendapatkan hasil prediksi "
    "dari model Machine Learning."
)

st.divider()

# Form input
with st.form("prediction_form"):
    st.subheader("Input Data")

    age = st.number_input("Age", value=30.0)
    sex = st.number_input("Sex", value=0.0)
    bmi = st.number_input("BMI", value=0.0)
    bp = st.number_input("Blood Pressure", value=0.0)

    s1 = st.number_input("S1", value=0.0)
    s2 = st.number_input("S2", value=0.0)
    s3 = st.number_input("S3", value=0.0)
    s4 = st.number_input("S4", value=0.0)
    s5 = st.number_input("S5", value=0.0)
    s6 = st.number_input("S6", value=0.0)

    submitted = st.form_submit_button(
        "Prediksi",
        use_container_width=True
    )

# Proses prediksi
if submitted:
    payload = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "bp": bp,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "s5": s5,
        "s6": s6
    }

    try:
        with st.spinner("Memproses prediksi..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )

        if response.ok:
            result = response.json()

            st.success("Prediksi berhasil!")

            st.subheader("Hasil Prediksi")
            st.json(result)

        else:
            st.error(
                f"API mengembalikan status "
                f"{response.status_code}"
            )
            st.code(response.text)

    except requests.exceptions.Timeout:
        st.error("Permintaan API melewati batas waktu.")

    except requests.exceptions.ConnectionError:
        st.error(
            "Tidak dapat terhubung ke API. "
            "Periksa URL dan status deployment."
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Terjadi kesalahan: {e}")
