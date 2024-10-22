import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_msg = st.text_input("enter the message")

def transform_text(text):
    #convert the whole text to lower case
    text = text.lower()
    #tokenization of the converted text
    text = nltk.word_tokenize(text)
    #remove the intended SPECIAL CHARACTERS in the text, since they actually don't have any meaning in the prediction.
    y= []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    # Remove stop words and punctuation marks.
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    #stemming:
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

#1. Data preprocessing
transformed_msg = transform_text(input_msg)

#2. Vectorize
vector_input = tfidf.transform([transformed_msg])

#3. predict
result = model.predict(vector_input)[0]

#4. Display
if result == 1:
    st.header("spam")
else:
    st.header("Not Spam")

