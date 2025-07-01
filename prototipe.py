import streamlit as st
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt

USERNAME = "admincs"
PASSWORD = "adorable123"

def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful! ✅")
            st.rerun()
        else:
            st.error("Username atau Password salah!")

def home():
    st.header("🏠 Home")
    st.write("Selamat datang di aplikasi analisis!")
      # Penjelasan tentang aplikasi
    st.write("""
    Aplikasi ini dapat melakukan analisis data secara interaktif dan visual. Berikut adalah panduan singkat untuk memulai:

    1. Unggah dataset dengan memilih opsi 'Upload Data' di menu.
    2. Pilih 'Exploratory Data Analysis' untuk melihat gambaran umum dari data melalui grafik dan statistik deskriptif.
    3. Pilih 'Analisis Sentimen' untuk menganalisis teks dalam dataset dan melihat sentimen yang terkandung di dalamnya.
    4. Pilih 'Analisis Topik' untuk mengidentifikasi topik-topik utama yang terkandung dalam dataset.
    5. Setelah analisis selesai, hasil analisis dalam bentuk grafik atau tabel dapat diunduh.
    """)


def upload_data():
    st.header("📤 Upload Data Excel")

    uploaded_file = st.file_uploader("Upload file Excel", type="xlsx")

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Gagal membaca file Excel: {e}")
            return

        if df.empty or df.columns.size == 0:
            st.error("❌ File kosong atau tidak memiliki kolom.")
            return

        # Simpan ke session_state
        st.session_state.df = df

        st.success("✅ Data berhasil diupload!")

    # Informasi dataset
    st.subheader("📈 Informasi Dataset")
    st.write(f"🔢 Jumlah baris: {df.shape[0]}")
    st.write(f"🔢 Jumlah kolom: {df.shape[1]}")

    # Cek missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        st.write("❗ Missing values per kolom:")
        st.dataframe(missing_cols)
    else:
        st.write("✅ Tidak ada missing values.")

    # Cek duplikat data
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        st.write(f"📌 Jumlah baris duplikat (semua kolom sama persis): {dup_count}")
        if st.checkbox("Tampilkan baris duplikat"):
            st.dataframe(df[df.duplicated()])
    else:
        st.write("✅ Tidak ada baris duplikat.")

    # Tipe data
    st.write("🔠 Tipe Data per Kolom:")
    st.dataframe(df.dtypes.astype(str))

    #  Tampilkan Dataset
    st.write("🔍 10 Data Teratas")
    st.dataframe(df.head(10))

    # Upload kamus slang dan stopwords
    st.subheader("📚 Upload Kamus Slang dan Stopwords")

    slang_file = st.file_uploader("Upload Kamus Slang (.xlsx)", type="xlsx", key="slang")
    stopwords_file = st.file_uploader("Upload Stopwords (.xlsx)", type="xlsx", key="stopwords")

    if slang_file and stopwords_file:
        kamus_slang_df = pd.read_excel(slang_file)
        stopwords_df = pd.read_excel(stopwords_file)

        kamus_slang = dict(zip(kamus_slang_df["slang"], kamus_slang_df["formal"]))
        list_stopwords = set(stopwords_df["stopword"])
        kata_hapus = {'nya', 'ya', 'sih', 'banget', 'gitu', 'deh', 'huhu', 'sayang', 'kali', 'wkwk', 'eh', 'ku', 'kak', 'adorable', 'sepatu', 'pakai', 'sih', 'dah', 'moga', 'semoga', 'x', 'projects', 'beli', 'pokok'}

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        # Step 1 - cleaning karakter, huruf berulang, spasi
        def clean_text(text):
            text = re.sub(r'[^a-z\s]', '', str(text), flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
            return text.lower()

        # Step 2 - ganti slang
        def replace_slang(text):
            words = text.split()
            return ' '.join([kamus_slang.get(w, w) for w in words])

        # Step 3 - hapus stopwords
        def remove_stopwords(text):
            words = text.split()
            return ' '.join([w for w in words if w not in list_stopwords])

        # Step 4 - stemming
        def apply_stemming(text):
            return stemmer.stem(text)

        # Step 5 - hapus noise
        def remove_noise(text):
            words = text.split()
            return ' '.join([w for w in words if w not in kata_hapus])

        # Step 6 - tokenisasi
        def tokenize(text):
            return text.split()

        # Kolom target
        if "Ulasan" not in df.columns:
            st.error("Kolom 'Ulasan' tidak ditemukan dalam data.")
            return

        st.write("🔄 Memproses pembersihan teks...")

        df["Ulasan_Cleaned"] = df["Ulasan"].apply(clean_text)
        df["Ulasan_Normalized"] = df["Ulasan_Cleaned"].apply(replace_slang)
        df["Ulasan_Removed"] = df["Ulasan_Normalized"].apply(remove_stopwords)
        df["Ulasan_Stemmed"] = df["Ulasan_Removed"].apply(apply_stemming)
        df["Ulasan_Stemmed2"] = df["Ulasan_Stemmed"].apply(remove_noise)
        df["Ulasan_Tokenized"] = df["Ulasan_Stemmed2"].apply(tokenize)

        st.success("✅ Teks berhasil dibersihkan!")

        # Tampilkan hasil
        st.subheader("🧼 Contoh Hasil Pembersihan")
        st.dataframe(df["Ulasan_Tokenized"].head())


def exploratory_data_analysis():
    st.header("📊 Exploratory Data Analysis")

    if "df" not in st.session_state or "Ulasan_Tokenized" not in st.session_state.df.columns:
        st.warning("⚠️ Silakan upload dan bersihkan data terlebih dahulu.")
        return

    df = st.session_state.df.copy()

    df["Ulasan_String"] = df["Ulasan_Tokenized"].apply(lambda tokens: ' '.join(tokens))

    # Explode kata
    words_exploded = df["Ulasan_String"].str.split().explode()
    word_counts = words_exploded.value_counts()

    # Plot histogram
    st.subheader("Kata yang paling banyak muncul")
    fig, ax = plt.subplots(figsize=(10, 6))
    word_counts.head(20).plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("Top 20 Most Frequent Words")
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def analisis_sentimen():
    st.header("💬 Analisis Sentimen")
    st.write("Halaman untuk analisis sentimen.")

def analisis_topik():
    st.header("📚 Analisis Topik")
    st.write("Halaman untuk analisis topik.")

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()
    else:
        st.sidebar.title("📂 Navigasi")
        page = st.sidebar.radio("Pilih Halaman", [
            "Home", "Upload Data", "Exploratory Data Analysis",
            "Analisis Sentimen", "Analisis Topik", "Logout"
        ])

        if page == "Home":
            home()
        elif page == "Upload Data":
            upload_data()
        elif page == "Exploratory Data Analysis":
            exploratory_data_analysis()
        elif page == "Analisis Sentimen":
            analisis_sentimen()
        elif page == "Analisis Topik":
            analisis_topik()
        elif page == "Logout":
            st.session_state.logged_in = False
            st.success("Logout berhasil.")
            st.experimental_rerun()

if __name__ == "__main__":
    main()
