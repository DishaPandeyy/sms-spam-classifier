import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    cleaned_text = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english'):
            cleaned_text.append(ps.stem(i))
    return " ".join(cleaned_text)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_text = st.text_input("Enter the message")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_text)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("Message is Spam")
    else:
        st.header("Message is Not Spam")