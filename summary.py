import os
import torch # type: ignore
import transformers # type: ignore
from transformers import pipeline  # type: ignore
import streamlit as st # type: ignore
from dotenv import load_dotenv # type: ignore
import streamlit as st # type: ignore

load_dotenv()

# global variables
model_name = os.getenv('MODEL_NAME')
task = os.getenv('TASK')

# torch cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# huggingface pipeline
summarizer = pipeline(task, model_name, framework='pt', device=device)


# summary generator
def generate_summary(text : str):
    response = summarizer(text, max_length = 100, min_length = 30)
    summary = response[0]['summary_text']
    return summary



# Streamlit App 
st.title('Summary Generator')
st.subheader('Generate summary of any text you want')

text = st.text_area(label = 'Text to analyse', placeholder = 'Write something...', max_chars = 2000)

clicked = st.button('Generate Summary')

progress_text = 'Generating summary...'

if clicked:
    if text=="":
        st.caption('Pls provide some textual input')
    else :
        summary = generate_summary(text)
        st.caption('This is your summary')
        st.write(summary)
