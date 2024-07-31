import streamlit as st
import requests
import json
from datetime import datetime
import pytz

# Function to generate and store Bible reading
def generate_and_store_bible_reading(api_key):
    date = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
    prompt = f"""
    Generate a Bible reading for the date {date}. Take into account key dates in the Protestant Christian calendar such as Christmas, Good Friday, Easter, and Easter Sunday.
    
    Provide the passage text in English.
    Offer a concise interpretation (50-100 words) that:
    - Explains the main message or theme
    - Highlights key lessons or applications
    - Connects the passage to modern life
    
    Include a short reflection question to encourage personal application.
    Add a brief historical or cultural context note (1-2 sentences) if relevant.
    Suggest a related Bible verse for further study.
    Conclude with a short prayer (2-3 sentences) inspired by the passage.
    
    Ensure the content is accessible to a general audience while maintaining theological accuracy.
    """

    response = requests.post(
        'https://api.openai.com/v1/engines/text-davinci-003/completions',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        data=json.dumps({
            'prompt': prompt,
            'max_tokens': 500,
        })
    )

    if response.status_code != 200:
        st.error("Error fetching the Bible reading.")
        return None

    data = response.json()
    if 'choices' in data and len(data['choices']) > 0:
        return data['choices'][0]['text']
    else:
        st.error("Failed to generate Bible reading.")
        return None

# Function to get or generate the daily Bible reading
def get_daily_bible_reading(api_key):
    today = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
    try:
        with open('bible_reading.json', 'r') as f:
            data = json.load(f)
            if data['date'] == today:
                return data['reading']
    except FileNotFoundError:
        pass

    reading = generate_and_store_bible_reading(api_key)
    if reading:
        with open('bible_reading.json', 'w') as f:
            json.dump({'date': today, 'reading': reading}, f)
        return reading
    else:
        return None

# Streamlit interface
st.title("Daily Bible Reading")

api_key = st.text_input("Enter your OpenAI API Key", type="password")

if api_key:
    reading = get_daily_bible_reading(api_key)
    if reading:
        st.markdown(reading.replace('\n', '\n\n'))
    else:
        st.error("Could not generate the Bible reading. Please check your API key and try again.")
