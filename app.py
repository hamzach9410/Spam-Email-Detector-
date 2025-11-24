import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Page configuration
st.set_page_config(
    page_title="🛡️ Spam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Ensure readable text colors */
    .stMarkdown, .stText {
        color: #2c3e50 !important;
    }
    
    /* Main header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #e8f4fd !important;
        font-size: 1.2rem;
    }
    
    /* Feature boxes */
    .feature-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease;
    }
    .feature-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
    }
    .feature-box h4 {
        color: #2c3e50 !important;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .feature-box p {
        color: #5a6c7d !important;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Result boxes */
    .result-spam {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.3);
    }
    .result-safe {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(81, 207, 102, 0.3);
    }
    
    /* Info section */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #3498db;
    }
    .info-box strong {
        color: #2c3e50 !important;
        font-size: 1.1rem;
    }
    .info-box {
        color: #5a6c7d !important;
        line-height: 1.6;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        border-radius: 12px;
        margin-top: 3rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .footer p {
        color: white !important;
        margin: 0.5rem 0;
    }
    .footer strong {
        color: #3498db !important;
    }
    
    /* Section headers */
    h3 {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #bdc3c7, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Set NLTK data path to a writable directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download required NLTK data with better error handling
@st.cache_resource
def download_nltk_data():
    try:
        # Set NLTK data path
        nltk.data.path.append(nltk_data_dir)
        
        # Download punkt tokenizer
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        # Download stopwords
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

# Download NLTK data
download_nltk_data()

ps=PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

# Main header with custom styling
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI-Powered Spam Detection System</h1>
    <p>Advanced Machine Learning Model for Email & SMS Classification</p>
</div>
""", unsafe_allow_html=True)

# Features section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h4>🚀 Real-time Detection</h4>
        <p>Instant spam classification</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h4>🎯 High Accuracy</h4>
        <p>Advanced NLP algorithms</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h4>🧠 Smart Analysis</h4>
        <p>Deep text processing</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main input section
st.markdown("### 📝 Enter Your Message for Analysis")
st.markdown("*Paste any email or SMS content below to check if it's spam or legitimate*")

input_sms = st.text_area(
    "Message Content:",
    placeholder="Enter your email or SMS content here...",
    height=150,
    help="Paste the complete message content for accurate analysis"
)

# Analysis button
if st.button("🔍 Analyze Message", use_container_width=True, type="primary"):
    if input_sms.strip():
        with st.spinner('🤖 Analyzing message...'):
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            
            # 4. Display results with custom styling
            if result == 1:
                st.markdown("""
                <div class="result-spam">
                    <h3>🚨 SPAM DETECTED!</h3>
                    <p>This message appears to be spam. Please be cautious!</p>
                </div>
                """, unsafe_allow_html=True)
                st.error("⚠️ Warning: This message has been classified as spam")
            else:
                st.markdown("""
                <div class="result-safe">
                    <h3>✅ LEGITIMATE MESSAGE</h3>
                    <p>This message appears to be safe and legitimate.</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ Safe: This message is not spam")
    else:
        st.warning("⚠️ Please enter a message to analyze")

# Information section
st.markdown("---")
st.markdown("### 📊 How It Works")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-box">
        <strong>🔧 Technology Stack:</strong><br><br>
        • Machine Learning Algorithm<br>
        • Natural Language Processing<br>
        • TF-IDF Vectorization<br>
        • Text Preprocessing Pipeline
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <strong>📈 Model Performance:</strong><br><br>
        • High accuracy classification<br>
        • Real-time processing<br>
        • Advanced spam detection<br>
        • Continuous learning
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Made with ❤️ by <strong>Ali Hamza</strong></p>
    <p>📧 Contact: ihamzaali@gmail.com</p>
    <p><em>Powered by Machine Learning & Natural Language Processing</em></p>
</div>
""", unsafe_allow_html=True)
