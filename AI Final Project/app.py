from flask import Flask
from flask import render_template
from flask import request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Ensure that the necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load the pre-trained model and vectorizer
tfidf = pickle.load(open('./models/vectorizer.pkl', 'rb'))
model = pickle.load(open('./models/model.pkl', 'rb'))

ps = PorterStemmer()

# Preprocess the text
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

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    input_sms = ""
    if request.method == 'POST':
        input_sms = request.form['message']
        # 1. Preprocess the text
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize the text
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict the result
        result = model.predict(vector_input)[0]
        # 4. Display the result
        result = "Fake" if result == 1 else "Real"
    
    return render_template('index.html', result=result, input_sms=input_sms)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)