import streamlit as st
import os
from textblob import TextBlob 
import speech_recognition as sr
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import PorterStemmer
from textblob import TextBlob 

port_stemmer = PorterStemmer()

# Create function for saving the uploaded file
def save_uploaded_file(uploaded_file, target_path):
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

# Create a function to calculate Sentiment scores for each text
def generate_polarity(text):
    sentiment = TextBlob(text).sentiment
    return sentiment

# Create a function to generate cleaned data from raw text
def clean_text(text):
    text = word_tokenize(text) # Create tokens
    text= " ".join(text) # Join tokens
    text = [char for char in text if char not in string.punctuation] # Remove punctuations
    text = ''.join(text) # Join the leters
    text = [char for char in text if char not in re.findall(r"[0-9]", text)] # Remove Numbers
    text = ''.join(text) # Join the leters
    text = [word.lower() for word in text.split() if word.lower() not in set(stopwords.words('english'))] # Remove common english words (I, you, we,...)
    text = ' '.join(text) # Join the leters
    text = list(map(lambda x: port_stemmer.stem(x), text.split()))
    return " ".join(text)   # error word

st.title('Sentimental Analyazer')

uploaded_file = st.file_uploader("Upload a file", type = ['mp3', 'wav'])

input_sms = st.text_input("Enter the Message")

if uploaded_file is not None:

    st.write("File uploaded successfully!")

    # Save the uploaded file
    save_uploaded_file(uploaded_file, os.path.join(r"C:\Users\kisha\Projects\Sentimental Analysis", "new.wav"))
    st.success("File saved successfully!")
    
    # Create a recognizer object
    recognizer = sr.Recognizer()

    # audio_path = "C:\\Users\\kisha\\Projects\\Sentimental Analysis\\new.wav"

    # Load the audio file
    with sr.AudioFile("new.wav") as source:
        # Read the entire audio file
        audio = recognizer.record(source)

    # Perform speech recognition
    try:
        print("Recognizing...")
        input_sms = recognizer.recognize_google(audio)  # Use Google Speech Recognition API
        print(f"You said: {input_sms}")
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    # Write Audio Text
    st.write(input_sms)

if st.button('Predict'):

    if input_sms == "" and uploaded_file is None:
        st.header('Please Enter Your Message !!!')

    else:
        input_sms = clean_text(input_sms)
        sentiment = generate_polarity(input_sms)
        st.header(sentiment[0])


# if __name__ == "__main__":
#     main()
