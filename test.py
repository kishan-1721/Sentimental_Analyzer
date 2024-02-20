import streamlit as st
import os
import speech_recognition as sr
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob

nltk.download('stopwords')
nltk.download('punkt')

class TextProcessor:
    def __init__(self):
        self.port_stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        text = word_tokenize(text) # Create tokens
        text = " ".join(text) # Join tokens
        text = [char for char in text if char not in string.punctuation] # Remove punctuations
        text = ''.join(text) # Join the letters
        text = [char for char in text if char not in re.findall(r"[0-9]", text)] # Remove Numbers
        text = ''.join(text) # Join the letters
        text = [word.lower() for word in text.split() if word.lower() not in self.stop_words] # Remove common english words (I, you, we,...)
        text = ' '.join(text) # Join the letters
        text = list(map(lambda x: self.port_stemmer.stem(x), text.split()))
        return " ".join(text)

class SentimentAnalyzer:
    @staticmethod
    def generate_polarity(text):
        sentiment = TextBlob(text).sentiment
        return sentiment

class FileHandler:
    @staticmethod
    def save_uploaded_file(uploaded_file, target_path):
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

def main():
    st.title('Sentimental Analyzer')

    uploaded_file = st.file_uploader("Upload a file", type=['mp3', 'wav'])
    input_sms = st.text_input("Enter the Message")

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        FileHandler.save_uploaded_file(uploaded_file, os.path.join( "new.wav"))
        st.success("File saved successfully!")

        recognizer = sr.Recognizer()

        with sr.AudioFile("new.wav") as source:
            audio = recognizer.record(source)

        try:
            print("Recognizing...")
            input_sms = recognizer.recognize_google(audio)
            print(f"You said: {input_sms}")
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

        st.write(input_sms)

    if st.button('Predict'):
        if input_sms == "" and uploaded_file is None:
            st.header('Please Enter Your Message !!!')
        else:
            text_processor = TextProcessor()
            input_sms = text_processor.clean_text(input_sms)
            sentiment = SentimentAnalyzer.generate_polarity(input_sms)
            st.header(sentiment[0])

if __name__ == "__main__":
    main()
