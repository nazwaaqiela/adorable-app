import streamlit as st
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import ngrams
import pickle
import io

USERNAME = "admincs"
PASSWORD = "adorable123"

def login():
    st.title("Halaman Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Berhasil masuk!")
            st.rerun()
        else:
            st.error("Username atau Password salah!")

def home():
    st.header("Halaman Utama")
    st.write("Selamat datang di aplikasi analisis data ulasan!")
      # Penjelasan tentang aplikasi
    st.write("""
    Aplikasi ini dapat melakukan analisis data secara interaktif dan visual. Berikut adalah panduan singkat untuk memulai:

    1. Unggah dataset dengan memilih opsi 'Unggah Data' di menu.
    2. Pilih 'Analisis Data Eksploratori' untuk melihat gambaran umum dari data melalui grafik dan statistik deskriptif.
    3. Pilih 'Analisis Sentimen' untuk menganalisis teks dalam dataset dan melihat sentimen yang terkandung di dalamnya.
    4. Pilih 'Analisis Topik' untuk mengidentifikasi topik-topik utama yang terkandung dalam dataset.
    5. Setelah analisis selesai, hasil analisis dalam bentuk grafik atau tabel dapat diunduh.
    """)


def upload_data():
    st.header("Unggah Data")

    uploaded_file = st.file_uploader("Unggah file Excel", type="xlsx")

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file Excel: {e}")
            return

        if df.empty or df.columns.size == 0:
            st.error("File kosong atau tidak memiliki kolom.")
            return

        # Simpan ke session_state
        st.session_state.df = df

        st.success("Data berhasil diunggah!")

    # Informasi dataset
    st.subheader("Informasi Dataset")
    st.write(f"Jumlah baris: {df.shape[0]}")
    st.write(f"Jumlah kolom: {df.shape[1]}")

    # Cek missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        st.write("Missing values per kolom:")
        st.dataframe(missing_cols)
    else:
        st.write("Tidak ada missing values.")

    # Cek duplikat data
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        st.write(f"Jumlah baris duplikat: {dup_count}")
        if st.checkbox("Tampilkan baris duplikat"):
            st.dataframe(df[df.duplicated()])
    else:
        st.write("Tidak ada baris duplikat.")

    # Tipe data
    st.write("Tipe Data per Kolom:")
    st.dataframe(df.dtypes.astype(str))

    #  Tampilkan Dataset
    st.write("10 Data Teratas")
    st.dataframe(df.head(10))

    # Upload kamus slang dan stopwords
    st.subheader("Unggah Kamus Slang dan Stopwords")

    slang_file = st.file_uploader("Unggah Kamus Slang (.xlsx)", type="xlsx", key="slang")
    stopwords_file = st.file_uploader("Unggah Stopwords (.xlsx)", type="xlsx", key="stopwords")

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

        if "Ulasan" not in df.columns:
            st.error("Kolom 'Ulasan' tidak ditemukan dalam data.")
            return

        status_placeholder = st.empty()
        status_placeholder.write("Memproses pembersihan teks...")

        df["Ulasan_Cleaned"] = df["Ulasan"].apply(clean_text)
        df["Ulasan_Normalized"] = df["Ulasan_Cleaned"].apply(replace_slang)
        df["Ulasan_Removed"] = df["Ulasan_Normalized"].apply(remove_stopwords)
        df["Ulasan_Stemmed"] = df["Ulasan_Removed"].apply(apply_stemming)
        df["Ulasan_Stemmed2"] = df["Ulasan_Stemmed"].apply(remove_noise)
        df["Ulasan_Tokenized"] = df["Ulasan_Stemmed2"].apply(tokenize)

        status_placeholder.empty()
        st.success("Teks berhasil dibersihkan!")

        # Tampilkan hasil
        st.subheader("Cuplikan Hasil Pembersihan")
        st.dataframe(df["Ulasan_Tokenized"].head())


def exploratory_data_analysis():
    st.header("Analisis Data Eksploratori")

    if "df" not in st.session_state or "Ulasan_Tokenized" not in st.session_state.df.columns:
        st.warning("⚠ Silakan upload dan bersihkan data terlebih dahulu.")
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

    # Gabungkan semua token jadi satu string
    all_tokens = sum(df["Ulasan_Tokenized"], [])
    text = ' '.join(all_tokens)

    # Buat WordCloud
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Tampilkan di Streamlit
    st.subheader("WordCloud")
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("WordCloud Ulasan", fontsize=20)
    st.pyplot(fig)

def analisis_sentimen():
    st.header("Analisis Sentimen Ulasan")

    if "df" not in st.session_state or "Ulasan_Tokenized" not in st.session_state.df.columns:
        st.warning("⚠ Silakan upload dan bersihkan data terlebih dahulu.")
        return

    df = st.session_state.df.copy()

    # Gabungkan token jadi string ulasan
    df["Ulasan_String"] = df["Ulasan_Tokenized"].apply(lambda tokens: ' '.join(tokens))

    # Load model pipeline
    try:
        with open("model_sentimen.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return

    # Prediksi sentimen
    try:
        df["Prediksi_Sentimen"] = model.predict(df["Ulasan_String"])
        st.success("Sentimen berhasil diprediksi!")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
        return

    label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
    export_df = df.copy()
    export_df["Sentimen"] = export_df["Prediksi_Sentimen"].map(label_map)

    st.subheader("Distribusi Sentimen")

    sentimen_counts = export_df["Sentimen"].value_counts().reindex(["Negatif", "Netral", "Positif"], fill_value=0)
    colors = {"Negatif": "red", "Netral": "gold", "Positif": "green"}

    fig, ax = plt.subplots(figsize=(5, 3.5))  
    bars = ax.bar(sentimen_counts.index, sentimen_counts.values,
                  color=[colors[s] for s in sentimen_counts.index])

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)  

    ax.set_title("Jumlah Ulasan per Sentimen", fontsize=8)
    ax.set_xlabel("Sentimen", fontsize=7)
    ax.set_ylabel("Jumlah", fontsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    st.pyplot(fig)
    
    st.subheader("Ulasan Berdasarkan Sentimen")

    tab_neg, tab_net, tab_pos = st.tabs(["**Negatif**", "**Netral**", "**Positif**"])

    with tab_neg:
        st.write("Ulasan dengan sentimen **Negatif**")
        neg_df = df[df["Prediksi_Sentimen"] == 0]
        st.dataframe(neg_df[["Ulasan"]])

        if not neg_df.empty:
            all_tokens = sum(neg_df["Ulasan_Tokenized"], [])
            text = ' '.join(all_tokens)
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("WordCloud Sentimen Negatif", fontsize=18)
            st.pyplot(fig)

    with tab_net:
        st.write("Ulasan dengan sentimen **Netral**")
        net_df = df[df["Prediksi_Sentimen"] == 1]
        st.dataframe(net_df[["Ulasan"]])

        if not net_df.empty:
            all_tokens = sum(net_df["Ulasan_Tokenized"], [])
            text = ' '.join(all_tokens)
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("WordCloud Sentimen Netral", fontsize=18)
            st.pyplot(fig)

    with tab_pos:
        st.write("Ulasan dengan sentimen **Positif**")
        pos_df = df[df["Prediksi_Sentimen"] == 2]
        st.dataframe(pos_df[["Ulasan"]])

        if not pos_df.empty:
            all_tokens = sum(pos_df["Ulasan_Tokenized"], [])
            text = ' '.join(all_tokens)
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("WordCloud Sentimen Positif", fontsize=18)
            st.pyplot(fig)

    kolom_terpilih = ["No", "Tanggal", "Produk", "Ulasan", "Ulasan_Tokenized", "Sentimen"]
    export_df = export_df[kolom_terpilih]
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name="Hasil Sentimen")

    xlsx_data = output.getvalue()  

    # Unduh di Streamlit
    st.download_button(
        label="📥 Unduh Hasil Sentimen",
        data=xlsx_data,
        file_name="hasil_sentimen.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Simpan kembali ke session_state
    st.session_state.df = df

def analisis_topik():
    st.header("Analisis Topik Ulasan")
  
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()
    else:
        st.sidebar.title("Navigasi")
        page = st.sidebar.radio("Pilih Halaman", [
            "Halaman Utama", "Unggah Data", "Analisis Data Eksploratori",
            "Analisis Sentimen", "Analisis Topik", "Keluar"
        ])

        if page == "Halaman Utama":
            home()
        elif page == "Unggah Data":
            upload_data()
        elif page == "Analisis Data Eksploratori":
            exploratory_data_analysis()
        elif page == "Analisis Sentimen":
            analisis_sentimen()
        elif page == "Analisis Topik":
            analisis_topik()
        elif page == "Keluar":
            st.session_state.logged_in = False
            st.success("Logout berhasil.")
            st.experimental_rerun()

if __name__ == "__main__":
    main()
