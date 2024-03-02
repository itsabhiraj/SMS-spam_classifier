import streamlit as st
import pickle

tf = pickle.load(open('vectorizer.pkl','rb'))
mnb = pickle.load(open('model.pkl', 'rb'))
sms_transf = pickle.load(open('transf.pkl', 'rb'))

st.title("SMS spam classifier")

input_sms = st.text_input("Enter the message")

trans_sms = sms_transf(input_sms)

vector_input = tf.transform([trans_sms])

result = mnb.predict(vector_input)[0]

if result == 1 :
    st.header('spam')

else:
    st.header('not spam')