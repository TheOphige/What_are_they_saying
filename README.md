# What are they saying?

## Project Overview
This project translates, summarizes, and provides insights into any text or Wikipedia page content in your preferred language. It generates a concise summary, extracts keywords, and visualizes them in an interactive word cloud.

## Features
- **Text Input or Wikipedia URL**: Analyze text or Wikipedia page content.
- **Language Translation**: Translate content to and from multiple languages.
- **Summarization**: Generate a concise summary of the translated text.
- **Keyword Extraction**: Extract important keywords from the summary.
- **Interactive Word Cloud**: Visualize keywords with interactive word clouds.

![Project Screenshot](assets/screenshot.png)

## Project Structure

Here’s the structure of the project:
```
what_are_they_saying/
├── data/
│   ├── raw/                 # Raw data (e.g., scraped Wikipedia pages)
│   ├── processed/           # Processed data (e.g., translated text, summaries)
│   ├── word_clouds/         # Generated word cloud images
│   └── keywords/            # Extracted keywords stored in JSON or CSV
├── notebooks/
│   ├── exploration.ipynb    # Jupyter notebooks for data exploration and prototyping
│   └── model_testing.ipynb  # Notebooks for testing models (e.g., translation, summarization)
├── src/
│   ├── __init__.py          # Makes src a Python package
│   ├── data_processing.py   # Functions for data processing (e.g., scraping, cleaning)
│   ├── translation.py       # Translation functions and API integrations
│   ├── summarization.py     # Text summarization functions
│   ├── keyword_extraction.py# Keyword extraction logic
│   ├── visualization.py     # Functions for generating and handling word clouds
│   ├── utils.py             # Utility functions (e.g., text pre-processing, language detection)
│   └── config.py            # Configuration settings (e.g., API keys, default settings)
├── streamlit_app/
│   ├── __init__.py          # Makes streamlit_app a Python package
│   ├── main.py              # Main Streamlit app script
│   ├── sidebar.py           # Streamlit sidebar code
│   ├── layout.py            # Layout configuration for the Streamlit app
│   ├── callbacks.py         # Callback functions for interactive elements
│   └── components.py        # Custom Streamlit components (e.g., word cloud visualization)
├── tests/
│   ├── __init__.py          # Makes tests a Python package
│   ├── test_translation.py  # Unit tests for translation functions
│   ├── test_summarization.py# Unit tests for summarization functions
│   ├── test_keywords.py     # Unit tests for keyword extraction
│   ├── test_visualization.py# Unit tests for word cloud generation
│   └── test_utils.py        # Unit tests for utility functions
├── .gitignore               # Git ignore file to exclude specific files and directories
├── requirements.txt         # List of dependencies (e.g., Streamlit, transformers, wordcloud)
├── README.md                # Project overview and instructions
├── Makefile                 # Automate common tasks (e.g., testing, running the app)
└── .env                     # Environment variables (e.g., API keys, settings)
```

## Installation
```bash
git clone https://github.com/TheOphige/what_are_they_saying.git
cd what_are_they_saying

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

make install
```

## Set up environment variables. 
Create a .env file in the project root with your Hugging Face API key if needed:
```
HUGGINGFACE_API_KEY = your_huggingface_api_key_here
```

## Running the App
```
make run
```

## Usage
How to use the application, including any options and functionality.

1. **Choose Input Type**: Select between "Text" or "Wikipedia URL".
2. **Enter Input**: Provide the text or URL to analyze.
3. **Select Languages**: Choose source and target languages for translation.
4. **Analyze**: Click the "Analyze" button to process the input.
5. **View Results**:
   - Summary of the translated text.
   - Keywords extracted from the summary.
   - Interactive word cloud displaying word frequencies and meanings.

For more details on the input and expected output, refer to the Streamlit app interface.

## Testing
```
make test
```

## Contributing
Guidelines for contributing to the project.

If you want to contribute to this project, please follow these guidelines:

1. Fork the repository and create a new branch for your changes.
2. Make your changes and add appropriate tests.
3. Submit a pull request with a description of your changes.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [E-mail](mailto:igetheophilus02@gmail.com).
