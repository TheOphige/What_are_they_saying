import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit.components.v1 as components
import json

def generate_word_cloud(keywords: list):
    """
    Generate an interactive word cloud from the extracted keywords and embed it in Streamlit.
    """
    # Create a dictionary of keywords and their frequencies
    word_freq = dict(keywords)
    
    # Generate the word cloud image
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    # Save the word cloud image to a file
    wordcloud_image_path = 'data/word_clouds/wordcloud.png'
    wordcloud.to_file(wordcloud_image_path)
    
    # Create a JSON representation of word frequencies for JavaScript
    word_freq_json = json.dumps(word_freq)
    
    # Embed the image and JavaScript for interactivity
    html_code = f"""
    <html>
    <head>
        <style>
            #wordcloud {{width: 800px; height: 400px;}}
            .tooltip {{
                position: absolute;
                display: none;
                padding: 5px;
                background: rgba(0, 0, 0, 0.7);
                color: #fff;
                border-radius: 3px;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <img id="wordcloud" src="{wordcloud_image_path}" alt="Word Cloud">
        <div id="tooltip" class="tooltip"></div>
        <script>
            const wordFreq = {word_freq_json};
            document.getElementById('wordcloud').addEventListener('mouseover', function(event) {{
                const tooltip = document.getElementById('tooltip');
                const word = event.target.alt;
                if (word && wordFreq[word]) {{
                    tooltip.style.left = `${{event.clientX + 10}}px`;
                    tooltip.style.top = `${{event.clientY + 10}}px`;
                    tooltip.style.display = 'block';
                    tooltip.innerText = 'Word: ' + word + ', Frequency: ' + wordFreq[word];
                }}
            }});
            document.getElementById('wordcloud').addEventListener('mouseout', function(event) {{
                const tooltip = document.getElementById('tooltip');
                tooltip.style.display = 'none';
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=450)
