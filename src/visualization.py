import streamlit.components.v1 as components
import json

def generate_word_cloud(keywords: list):
    """
    Generate an interactive word cloud from the extracted keywords and embed it in Streamlit.
    """
    # Create a dictionary of keywords and their frequencies
    word_freq = dict(keywords)

    # Convert the word frequency dictionary to a JSON object for JavaScript
    word_freq_json = json.dumps([{"text": word, "weight": freq} for word, freq in word_freq.items()])

    # Embed the word cloud and JavaScript for interactivity
    html_code = f"""
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/wordcloud2.js/1.0.6/wordcloud2.min.js"></script>
        <style>
            #wordcloud {{ width: 100%; height: 450px; position: relative; }}
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
        <div id="wordcloud"></div>
        <div id="tooltip" class="tooltip"></div>
        <script>
            const wordFreq = {word_freq_json};
            const tooltip = document.getElementById('tooltip');
            
            WordCloud(document.getElementById('wordcloud'), {{
                list: wordFreq.map(item => [item.text, item.weight]),
                gridSize: 5,  // Adjust grid size for denser packing
                weightFactor: function(size) {{ return size * 1.5; }},  // Scale the size dynamically
                color: '#000000',
                backgroundColor: '#ffffff',
                rotateRatio: 0.5,  // Control word rotation (0 = no rotation)
                minSize: 10,  // Minimum size of words
                hover: function(item, dimension, event) {{
                    if (item) {{
                        tooltip.style.left = `${{event.clientX + 10}}px`;
                        tooltip.style.top = `${{event.clientY + 10}}px`;
                        tooltip.style.display = 'block';
                        tooltip.innerText = 'Word: ' + item[0] + ', Frequency: ' + item[1];
                    }} else {{
                        tooltip.style.display = 'none';
                    }}
                }},
                click: function(item) {{
                    if (item) {{
                        alert('You clicked on: ' + item[0]);
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # Display the word cloud in Streamlit
    components.html(html_code, height=450)


# import streamlit as st

# # Example list of keywords with frequencies
# # Adjusted keywords: 
# keywords = [
#     ('generación', 8.56),
#     ('Demografía_y_Ciencias', 9.14),
#     ('Sociales', 12.03),
#     ('Ciencias_Sociales', 12.24),
#     ('perspectivas', 13.32),
#     ('artículo', 15.47),
#     ('compartiendo_experiencias', 15.64),
#     ('características_similares', 15.64),
#     ('Definición', 15.80),
#     ('concepto', 15.82),
#     ('personas_nacidas', 17.67),
#     ('experiencias_y_características', 17.78),
#     ('grupo', 18.10),
#     ('personas', 19.32),
#     ('siglo_XIX', 20.74),
#     ('cambios_sociales', 21.69),
#     ('concepto_de_generaciones', 21.73),
#     ('generaciones_sociales_obtuvo', 22.80),
#     ('sociales_obtuvo_prioridad', 22.80),
#     ('generación_familiar', 22.93)
# ]

# # Call the function to generate and display the word cloud
# st.title("Interactive Word Cloud")
# generate_word_cloud(keywords)


