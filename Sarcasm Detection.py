import streamlit as st
import pandas as pd
from textblob import TextBlob
import re
from datetime import datetime
import json
import os
# Tambahan (opsional: untuk model lebih baik nanti)
# from transformers import pipeline
# sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Sarkasme", layout="wide")

# CSS Custom
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .sarcasm {
        background-color: #ff6b6b;
        color: white;
    }
    .non-sarcasm {
        background-color: #51cf66;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# File untuk menyimpan riwayat
HISTORY_FILE = "sarcasm_history.json"

# Fungsi untuk load riwayat
def load_history():
    """Load riwayat dari file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

# Fungsi untuk save riwayat
def save_history(history):
    """Save riwayat ke file"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# Fungsi untuk tambah riwayat
def add_to_history(text, result, confidence):
    """Tambah entry ke riwayat"""
    history = load_history()
    history.insert(0, {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'text': text,
        'result': result,
        'confidence': f"{confidence:.2%}"
    })
    # Batasi riwayat hingga 100 entry
    history = history[:100]
    save_history(history)

# Header
st.markdown("<h1 class='main-header'>🎭 Deteksi Sarkasme (Bahasa Indonesia)</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Pengaturan")
# Fokus hanya Bahasa Indonesia
detection_mode = st.sidebar.radio("Pilih Mode:", ["Single Text", "Batch Upload", "📋 Riwayat"])

# Fungsi deteksi sarkasme (DIoptimalkan untuk Bahasa Indonesia)
def detect_sarcasm(text):
    """Heuristik deteksi sarkasme khusus Bahasa Indonesia."""
    if not isinstance(text, str):
        text = str(text)
    text_lower = text.lower().strip()

    # Pola sarkasme / frasa sarkastik Bahasa Indonesia
    sarcasm_patterns_id = [
        r"\bkeren banget\b",
        r"\bhebat banget\b",
        r"\bbagus sekali\b",
        r"\bmantap banget\b",
        r"\bterima kasih banyak\b",
        r"\bya pasti\b",
        r"\bola bagus\b",  # catch common sarcastic praise
        r"\bwah hebat\b",
        r"\bbagus banget\b",
        r"\bnggak nyangka banget\b",
        r"\bsiapa peduli\b",
        r"\bsiapa yang butuh\b",
        r"\bpadahal\b",
        r"\bmalah\b",
        r"\bjustru\b",
        r"\bya ampun\b",
        r"\bbener banget\b",
        r"\bkomplit banget\b",
        r"\boke banget\b",
        r"\".+\"|\'.+\'",  # penggunaan kutip untuk menandai sarkasme
        r"\.\.\.$",       # trailing ellipsis
        r"\bwah luar biasa\b",
        r"\bbener2\b|\bbener\-bener\b",
        r"\bpeka banget\b",
        r"\bsuka banget\b",
        r"\bmakasih ya\b|\bmakasih banget\b"
    ]

    # Kata-kata kontras / penanda sarkasme
    contrast_markers = ["padahal", "malah", "justru", "ternyata", "sedangkan"]

    # Kamus kata positif/negatif sederhana Bahasa Indonesia (untuk skor sentimen kasar)
    positive_words = ["bagus","hebat","keren","mantap","luar biasa","sukak","sangat","terima kasih","baik","kerenbanget","oke"]
    negative_words = ["bodoh","payah","buruk","menyedihkan","sial","bete","capek","susah","gagal","ngaco","parah"]

    # Hitung kemunculan kata positif/negatif
    words = re.findall(r"\w+", text_lower)
    total_words = max(1, len(words))
    pos_count = sum(1 for w in words if any(p in w for p in positive_words))
    neg_count = sum(1 for w in words if any(n in w for n in negative_words))
    sentiment_score = (pos_count - neg_count) / total_words  # [-1,1] kasar

    # Pendeteksian pola sarkasme
    pattern_match = any(re.search(p, text_lower) for p in sarcasm_patterns_id)
    has_contrast = any(m in text_lower for m in contrast_markers)
    has_quotes = bool(re.search(r"\".+\"|'.+'", text))
    has_ellipsis = text_lower.endswith("...") or text_lower.endswith("…")

    # Heuristik final:
    # - Jika ada pola eksplisit (pujian hiperbolik, kutipan, "siapa peduli", "padahal" dll) -> Sarkasme
    # - Jika ada pola + sentiment_score sangat positif -> lebih yakin
    # - Jika ada kontras atau kutip -> kemungkinan sarkasme
    confidence = 0.5
    if pattern_match:
        confidence = 0.75 + min(0.2, abs(sentiment_score))
        return "Sarkasme", min(0.98, confidence)
    if has_quotes or has_ellipsis or has_contrast:
        # jika ada tanda kutip/ellipsis/kontras dan sentiment bertentangan (positif padahal konteks negatif) -> Sarkasme
        if sentiment_score > 0.1:
            return "Sarkasme", 0.7 + min(0.18, sentiment_score)
        return "Kemungkinan Sarkasme", 0.55
    # fallback: jika sentiment negatif dominan dan tidak ada pola -> Bukan Sarkasme
    if sentiment_score < -0.15:
        return "Bukan Sarkasme", min(0.85, abs(sentiment_score))
    # default: bukan sarkasme
    return "Bukan Sarkasme", max(0.35, 0.35 + sentiment_score)

# Mode 1: Single Text
if detection_mode == "Single Text":
    st.subheader("📝 Masukkan Teks")
    user_input = st.text_area("Masukkan kalimat yang ingin dianalisis:", height=100)
    
    if st.button("🔍 Deteksi", use_container_width=True):
        if user_input.strip():
            result, confidence = detect_sarcasm(user_input)
            
            # Tambah ke riwayat
            add_to_history(user_input, result, confidence)
            
            # Tampilkan hasil
            if result == "Sarkasme":
                st.markdown(f"""
                    <div class='result-box sarcasm'>
                    <h3>✅ Hasil: {result}</h3>
                    <p><b>Confidence:</b> {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            elif result == "Kemungkinan Sarkasme":
                st.markdown(f"""
                    <div class='result-box' style='background-color: #ffd93d; color: black;'>
                    <h3>⚠️ Hasil: {result}</h3>
                    <p><b>Confidence:</b> {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='result-box non-sarcasm'>
                    <h3>✓ Hasil: {result}</h3>
                    <p><b>Confidence:</b> {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Analisis detail
            with st.expander("📊 Analisis Detail"):
                # tetap tampilkan skor sederhana
                st.write(f"**Sentiment (kasar):** {round((pos_count - neg_count) / max(1, len(words)), 3) if 'words' in locals() else 'N/A'}")
                try:
                    blob = TextBlob(user_input)
                    st.write(f"**Polarity (TextBlob):** {blob.sentiment.polarity:.3f}")
                    st.write(f"**Subjectivity (TextBlob):** {blob.sentiment.subjectivity:.3f}")
                except Exception:
                    pass
                st.write(f"**Jumlah Kata:** {len(user_input.split())}")
        else:
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")

# Mode 2: Batch Upload
elif detection_mode == "Batch Upload":
    st.subheader("📤 Upload File CSV")
    uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Preview Data:**")
        st.dataframe(df.head())
        
        # Pilih kolom
        text_column = st.selectbox("Pilih kolom teks:", df.columns)
        
        if st.button("🔍 Analisis Semua", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in df.iterrows():
                text = row[text_column]
                result, confidence = detect_sarcasm(str(text))
                results.append({
                    'Text': text,
                    'Hasil': result,
                    'Confidence': f"{confidence:.2%}"
                })
                progress_bar.progress((idx + 1) / len(df))
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Download hasil
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Hasil",
                data=csv,
                file_name="hasil_deteksi_sarkasme.csv",
                mime="text/csv"
            )
            
            # Statistik
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analisis", len(results_df))
            with col2:
                sarcasm_count = len(results_df[results_df['Hasil'] == 'Sarkasme'])
                st.metric("Sarkasme", sarcasm_count)
            with col3:
                non_sarcasm_count = len(results_df[results_df['Hasil'] == 'Bukan Sarkasme'])
                st.metric("Bukan Sarkasme", non_sarcasm_count)

# Mode 3: Riwayat
else:
    st.subheader("📋 Riwayat Deteksi")
    
    history = load_history()
    
    if not history:
        st.info("📭 Belum ada riwayat deteksi.")
    else:
        # Statistik riwayat
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Riwayat", len(history))
        with col2:
            sarcasm_count = sum(1 for h in history if h['result'] == 'Sarkasme')
            st.metric("Sarkasme", sarcasm_count)
        with col3:
            non_sarcasm_count = sum(1 for h in history if h['result'] == 'Bukan Sarkasme')
            st.metric("Bukan Sarkasme", non_sarcasm_count)
        with col4:
            maybe_count = sum(1 for h in history if h['result'] == 'Kemungkinan Sarkasme')
            st.metric("Kemungkinan", maybe_count)
        
        st.divider()
        
        # Search dan filter
        col_search, col_filter = st.columns([3, 1])
        with col_search:
            search_term = st.text_input("🔍 Cari riwayat:")
        with col_filter:
            filter_result = st.selectbox("Filter Hasil:", 
                                        ["Semua", "Sarkasme", "Bukan Sarkasme", "Kemungkinan Sarkasme"])
        
        # Filter riwayat
        filtered_history = history
        if search_term:
            filtered_history = [h for h in filtered_history if search_term.lower() in h['text'].lower()]
        if filter_result != "Semua":
            filtered_history = [h for h in filtered_history if h['result'] == filter_result]
        
        # Tampilkan riwayat dengan format yang lebih baik
        st.divider()
        st.write(f"**Menampilkan {len(filtered_history)} dari {len(history)} riwayat**")
        
        for idx, entry in enumerate(filtered_history):
            # Warna berdasarkan hasil
            if entry['result'] == 'Sarkasme':
                color = '#ff6b6b'
                emoji = '✅'
            elif entry['result'] == 'Kemungkinan Sarkasme':
                color = '#ffd93d'
                emoji = '⚠️'
            else:
                color = '#51cf66'
                emoji = '✓'
            
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown(f"""
                    <div style='background-color: {color}; color: white; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
                    <p style='margin: 0;'><b>{emoji} {entry['result']}</b> | Confidence: {entry['confidence']}</p>
                    <p style='margin: 5px 0 0 0; font-size: 14px;'>"{entry['text']}"</p>
                    <p style='margin: 5px 0 0 0; font-size: 12px; opacity: 0.8;'>{entry['timestamp']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"delete_{idx}", help="Hapus item"):
                    history.pop(history.index(entry) if entry in history else idx)
                    save_history(history)
                    st.rerun()
        
        st.divider()
        
        # Tombol hapus semua
        col1, col2 = st.columns(2)
        with col1:
            # Download riwayat
            history_df = pd.DataFrame(history)
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Riwayat",
                data=csv,
                file_name=f"riwayat_sarkasme_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("🗑️ Hapus Semua Riwayat", use_container_width=True, type="secondary"):
                if st.checkbox("Saya yakin ingin menghapus semua riwayat"):
                    save_history([])
                    st.success("✅ Semua riwayat telah dihapus!")
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("Dibuat dengan ❤️ menggunakan Streamlit")