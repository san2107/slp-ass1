import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model
model_name = 'bert-base-uncased'  # You can use other pre-trained BERT models as well

# Specify cache directory
cache_dir = "cache_directory"  # Replace "cache_directory" with the path to your cache directory
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = BertForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
model.eval()

def predict_sentiment(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt')
   
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
   
    # Interpret results
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class == 1:
        return "Positive sentiment"
    else:
        return "Negative sentiment"

# Streamlit app
st.title("BERT Sentiment Analysis")

text_input = st.text_area("Enter text for sentiment analysis:", "")

if st.button("Analyze"):
    if text_input.strip() != "":
        sentiment = predict_sentiment(text_input)
        st.write("Sentiment:", sentiment)
    else:
        st.write("Please enter some text for analysis.")





