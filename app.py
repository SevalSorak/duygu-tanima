import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Duygu Tanıma Sistemi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ana stil tanımlamaları
st.markdown(
    """
    <style>
    /* Genel sayfa stili */
    .main {
        background-color: #F8F9FA;
        padding: 0;
    }
    
    /* Başlık stili */
    .title-container {
        background: linear-gradient(90deg, #2C3E50, #4CA1AF);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: white !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    
    .subtitle {
        font-size: 1.2rem !important;
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 400 !important;
    }
    
    /* Bölüm stilleri */
    .section-title {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #2C3E50 !important;
        border-left: 6px solid #4CA1AF;
        padding-left: 15px;
        margin-bottom: 1.5rem !important;
        background-color: rgba(76, 161, 175, 0.1);
        padding: 12px 15px;
        border-radius: 0 8px 8px 0;
    }
    
    /* Grafik stili */
    .chart-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Tahmin kartı */
    .prediction-container {
        background: linear-gradient(135deg, #ffffff, #f5f7fa);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .prediction-emoji {
        font-size: 5rem;
        margin-bottom: 1rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #555;
        margin-bottom: 0.5rem;
    }
    
    .prediction-value {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #4CA1AF, #2C3E50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .confidence-label {
        font-size: 1.1rem;
        color: #777;
        font-weight: 500;
    }
    
    /* Bilgi kutusu */
    .info-box {
        background-color: rgba(76, 161, 175, 0.1);
        border-left: 5px solid #4CA1AF;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
    
    .stApp a {
        color: #4CA1AF !important;
    }
    
    /* Alt bölüm başlığı */
    .subsection-title {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #2C3E50 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid #4CA1AF;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    /* Hoşgeldiniz resmi için stil */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        text-align: center;
    }
    
    .welcome-image {
        max-width: 300px;
        margin-bottom: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .welcome-text {
        color: #2C3E50;
        font-size: 1.2rem;
        font-weight: 500;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar konfigürasyonu
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712108.png", width=100)
    st.title("Hakkında")
    st.info("""
    Bu uygulama, VGG16 modeli kullanarak yüz ifadelerinden duygu tanıma yapar.
    
    **Desteklenen duygular:**
    - 😠 Kızgın (Angry)
    - 😊 Mutlu (Happy)
    - 😢 Üzgün (Sad)
    - 😨 Korku (Fear)
    """)
    
    st.markdown("---")
    st.subheader("Kullanım Talimatları")
    st.markdown("""
    1. Bir yüz ifadesi görüntüsü yükleyin
    2. Model otomatik olarak duyguyu tanıyacaktır
    3. Sonuçlar görselleştirilecektir
    """)
    
    st.markdown("---")
    st.caption("© 2025 Duygu Tanıma Sistemi | VGG16 Mimarisi")

# Ana başlık
st.markdown(
    """
    <div class="title-container">
        <h1 class="main-title">🧠 Yapay Zeka Duygu Tanıma Sistemi</h1>
        <p class="subtitle">VGG16 derin öğrenme modeli ile yüz ifadelerinden duygu analizi</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Ana içerik düzeni
col1, col2 = st.columns([1, 1])

# Eğitilmiş modeli yükle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("VGG16.h5")
    return model

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("Model dosyası yüklenemedi. Lütfen 'VGG16.h5' dosyasının doğru konumda olduğundan emin olun.")

# Sınıf isimleri ve emojiler
class_names = ["angry", "happy", "sad", "fear"]
emojis = {
    'angry': '😠',
    'happy': '😊',
    'sad': '😢',
    'fear': '😨'
}

# Renk paleti
colors = ["#E74C3C", "#2ECC71", "#3498DB", "#9B59B6"]

# Görüntü yükleme bölümü
with col1:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📸 Görüntü Yükleme</h2>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="info-box">
            <strong>Desteklenen Formatlar:</strong> JPG, JPEG, PNG<br>
            <strong>İdeal Boyut:</strong> En az 48x48 piksel<br>
            <strong>En İyi Sonuç İçin:</strong> Net yüz ifadeleri içeren görüntüler yükleyin
        </div>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader("📤 **Bir yüz ifadesi görüntüsü yükleyin**", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Yüklenen Görüntü", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sonuçlar bölümü
with col2:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📊 Analiz Sonuçları</h2>', unsafe_allow_html=True)
    
    if uploaded_file is not None and model_loaded:
        with st.spinner("🔄 Görüntü işleniyor..."):
            img = Image.open(uploaded_file)
            img = img.resize((48, 48))
            img = img.convert("RGB")
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

        with st.spinner("🤖 Model tahmin yapıyor..."):
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)
            confidence = np.max(predictions)

        predicted_emotion = class_names[predicted_class[0]]
        emoji = emojis[predicted_emotion]
        
        st.markdown(f"""
            <div class="prediction-container">
                <div class="prediction-emoji">{emoji}</div>
                <div class="prediction-label">Tespit Edilen Duygu:</div>
                <div class="prediction-value">{predicted_emotion.upper()}</div>
                <div class="confidence-label">Güven Oranı: {confidence:.2%}</div>
            </div>
        """, unsafe_allow_html=True)        
        
    else:
        st.markdown(
            """
            <div class="welcome-container">
                <img src="https://cdn-icons-png.flaticon.com/512/10817/10817491.png" class="welcome-image">
                <p class="welcome-text">
                    <b>Yapay Zeka Duygu Tanıma Sistemine Hoş Geldiniz!</b><br>
                    Lütfen analiz için bir yüz ifadesi görüntüsü yükleyin.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Grafik bölümü 
if uploaded_file is not None and model_loaded:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📈 Görsel Analiz</h2>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plt.style.use('fivethirtyeight')
    ax.set_facecolor('#f5f5f5')
    fig.patch.set_facecolor('#f5f5f5')
    
    sorted_indices = np.argsort(predictions[0])[::-1]
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_values = [predictions[0][i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    bars = ax.bar(
        [f"{emojis[name]} {name.capitalize()}" for name in sorted_names],
        sorted_values,
        color=sorted_colors,
        edgecolor='white',
        linewidth=1.5,
        alpha=0.8,
        width=0.6
    )
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f"{height:.1%}",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            color='#333'
        )
    
    ax.set_ylim(0, max(predictions[0]) * 1.15)
    ax.set_ylabel('Olasılık', fontsize=14, fontweight='bold', color='#333')
    ax.set_title('Duygu Sınıfları Olasılık Dağılımı', fontsize=16, fontweight='bold', color='#333', pad=20)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.tick_params(axis='x', labelsize=12, colors='#333')
    ax.tick_params(axis='y', labelsize=12, colors='#333')
    
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.markdown(
        """
        <div class="info-box">
            <strong>Grafik Açıklaması:</strong> Yukarıdaki grafik, yüklenen yüz ifadesinde tespit edilen 
            duygu olasılıklarını göstermektedir. En yüksek çubuk, modelin en güvenli olduğu duygu sınıfını temsil eder.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Dipnot bölümü
st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f1f1f1; border-radius: 10px;">
        <p style="color: #777; font-size: 0.9rem;">
            Bu uygulama, TensorFlow ve VGG16 mimarisi kullanılarak geliştirilmiş bir duygu tanıma sistemini içermektedir.
            En iyi sonuçlar için net yüz ifadeleri içeren görüntüler kullanın.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Gizli model yükleme uyarısını engelle
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)