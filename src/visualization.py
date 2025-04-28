import streamlit as st
from wordcloud import WordCloud
import io
import requests
import os


# # Load environment variables from .env
# from dotenv import find_dotenv, load_dotenv

# # Load environment variables
# load_dotenv(find_dotenv())

# # Retrieve API keys from .env
# IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
IMGUR_CLIENT_ID = st.secrets["IMGUR_CLIENT_ID"]

# Upload image to Imgur
def upload_to_imgur(image):
    """Upload an image to Imgur and return the URL."""
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    url = "https://api.imgur.com/3/image"
    
    # Send the image to Imgur
    response = requests.post(url, headers=headers, files={"image": image})
    
    if response.status_code == 200:
        # Get the URL of the uploaded image
        image_url = response.json()["data"]["link"]
        return image_url
    else:
        st.error("Failed to upload image")
        return None

# Generate the word cloud
def generate_word_cloud(keywords: list):
    """
    Generate a colorful word cloud from the extracted keywords, upload it to Imgur, and display the image from Imgur.
    """
    # Create a dictionary of keywords and their frequencies
    word_freq = dict(keywords)
    
    # Generate word cloud using the wordcloud module with a colormap for colors
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        colormap='plasma'  # You can choose 'viridis', 'plasma', 'inferno', 'cividis', etc.
    ).generate_from_frequencies(word_freq)

    # Save the word cloud to an in-memory bytes buffer
    image_buffer = io.BytesIO()
    wordcloud_image = wordcloud.to_image()
    wordcloud_image.save(image_buffer, format='PNG')
    image_buffer.seek(0)  # Rewind to the start of the buffer
    
    # Upload the image to Imgur
    imgur_url = upload_to_imgur(image_buffer)
    
    if imgur_url:
        # Display the image from Imgur
        st.image(imgur_url, caption='Colorful Word Cloud', use_container_width=True)

