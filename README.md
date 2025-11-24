# 🛡️ AI-Powered Spam Detection System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/alihamza)
> A powerful machine learning-based spam detection system that can classify emails and SMS messages as spam or legitimate using advanced Natural Language Processing (NLP) techniques.

## 🌟 Features

- **🚀 Real-time Detection**: Instant classification of emails and SMS messages
- **🎯 High Accuracy**: Trained model with excellent performance metrics
- **💻 Beautiful UI**: Modern Streamlit web application with responsive design
- **🧠 Advanced NLP**: Sophisticated text preprocessing pipeline
- **📱 Mobile Friendly**: Works seamlessly on all devices
- **🛡️ Robust Error Handling**: Comprehensive error management

### 🔧 Technical Features

- **Text Preprocessing**: Tokenization, stop word removal, stemming, punctuation removal
- **Machine Learning**: TF-IDF vectorization with optimized classification algorithms
- **Web Framework**: Streamlit for interactive web interface
- **Model Persistence**: Pickle-based model serialization

## 🎯 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://email-spam-classifier-shazim.streamlit.app/)

Try the live demo: [Spam Classification App](https://email-spam-classifier-shazim.streamlit.app/)

## 📸 Screenshots

<div align="center">
  <img src="static/images/Streamlit - Personal - Microsoft​ Edge 31-Aug-25 10_51_49 PM.png" alt="App Screenshot" width="600"/>
  <p><em>Beautiful and intuitive user interface</em></p>
</div>

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alihamza/Email-spam-classifier.git
   cd Email-spam-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage

### Web Application

1. Start the application: `streamlit run app.py`
2. Open your browser to `http://localhost:8501`
3. Enter any email or SMS content in the text area
4. Click "Analyze Message" to get instant results
5. View color-coded results (🔴 Red for spam, 🟢 Green for legitimate)

### Machine Learning Pipeline

- **Algorithm**: [Your Model Type - e.g., Naive Bayes, SVM, Random Forest]
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: Processed text features
- **Output**: Binary classification (0 = Not Spam, 1 = Spam)

## 📊 Dataset

The model was trained on a comprehensive dataset containing:

- **📧 Spam Messages**: Various types of spam emails and SMS
- **✅ Legitimate Messages**: Normal, non-spam communications
- **📈 Data Source**: [kaggle.com]|

## 🔧 API Reference


## 🤝 Contributing

We love your input! We want to make contributing to this project as easy and transparent as possible.

## 📝 License

This project is for Learning & Educational Purpose

## 👨‍💻 Author

**Ali Hamza**

📧 **Email**: [ihamzaali@gmail.com](mailto:ihamzaali@gmail.com)

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/alihamza)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ihamzaali@gmail.com)

---

<div align="center">
  <p><strong>Made with ❤️ by Ali Hamza</strong></p>
  <p><em>🚀 Powered by Machine Learning & Natural Language Processing</em></p>
  <p>📧 Contact: ihamzaali@gmail.com</p>
</div>
