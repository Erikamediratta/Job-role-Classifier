from flask import Flask, render_template, request
import joblib
import pdfplumber
import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


model = joblib.load('resume_model.pkl')
tfidf = joblib.load('tfidf.pkl')

# Clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    return " ".join(cleaned_words)

def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    resume_file = request.files['resume']
  
    raw_text = extract_text(resume_file)
    cleaned = clean_text(raw_text)
    

    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0].max() * 100
    
    return render_template('index.html',
                           prediction=prediction,
                           probability=round(probability, 1))

if __name__ == '__main__':
    app.run(debug=True)