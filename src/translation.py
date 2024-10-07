import os
import requests
import time
import streamlit as st
from typing import Any, Dict
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/facebook/mbart-large-50-many-to-many-mmt"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

# List of supported languages with codes
SUPPORTED_LANGUAGES = {
    'ar': 'ar_AR', 'cs': 'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX', 
    'et': 'et_EE', 'fi': 'fi_FI', 'fr': 'fr_XX', 'gu': 'gu_IN', 'hi': 'hi_IN', 
    'it': 'it_IT', 'ja': 'ja_XX', 'kk': 'kk_KZ', 'ko': 'ko_KR', 'lt': 'lt_LT', 
    'lv': 'lv_LV', 'my': 'my_MM', 'ne': 'ne_NP', 'nl': 'nl_XX', 'ro': 'ro_RO', 
    'ru': 'ru_RU', 'si': 'si_LK', 'tr': 'tr_TR', 'vi': 'vi_VN', 'zh': 'zh_CN',
    'af': 'af_ZA', 'az': 'az_AZ', 'bn': 'bn_IN', 'fa': 'fa_IR', 'he': 'he_IL', 
    'hr': 'hr_HR', 'id': 'id_ID', 'ka': 'ka_GE', 'km': 'km_KH', 'mk': 'mk_MK', 
    'ml': 'ml_IN', 'mn': 'mn_MN', 'mr': 'mr_IN', 'pl': 'pl_PL', 'ps': 'ps_AF', 
    'pt': 'pt_XX', 'sv': 'sv_SE', 'sw': 'sw_KE', 'ta': 'ta_IN', 'te': 'te_IN', 
    'th': 'th_TH', 'tl': 'tl_XX', 'uk': 'uk_UA', 'ur': 'ur_PK', 'xh': 'xh_ZA',
    'gl': 'gl_ES', 'sl': 'sl_SI'
}

def translate_text(text: str, src_lang: str, tgt_lang: str, max_retries: int = 5, retry_delay: int = 10) -> Dict[str, Any]:
    """
    Translate text using the Hugging Face API with retry logic and language validation.

    Args:
        text (str): The input text to translate.
        src_lang (str): The source language code (e.g., 'ru' for Russian).
        tgt_lang (str): The target language code (e.g., 'en' for English).
        max_retries (int): Maximum number of retry attempts.
        retry_delay (int): Delay in seconds between retries.

    Returns:
        dict: The API response containing the translated text or an error message.
    
    Raises:
        RuntimeError: If the API request fails or an unsupported language is used.
    """
    # Validate source and target language codes
    if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language code(s). Supported languages are: {', '.join(SUPPORTED_LANGUAGES.keys())}")

    payload = {
        "inputs": text,
        "parameters": {
            "src_lang": SUPPORTED_LANGUAGES[src_lang],
            "tgt_lang": SUPPORTED_LANGUAGES[tgt_lang]
        }
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

            # Parse and return the JSON response
            return response.json()

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 503:
                # If the service is temporarily unavailable, retry
                retry_delay_seconds = response.json().get("estimated_time", retry_delay)
                st.info(f"Service unavailable. Retrying in {retry_delay_seconds} seconds... ({retries + 1}/{max_retries})")
                time.sleep(retry_delay_seconds)
                retries += 1

            elif response.status_code == 500 and 'Model too busy' in response.text:
                st.warning(f"Model too busy. Retrying in {retry_delay} seconds... ({retries + 1}/{max_retries})")
                time.sleep(retry_delay)
                retries += 1

            else:
                st.error(f"HTTP error occurred: {http_err}")
                break

        except Exception as err:
            st.error(f"An unexpected error occurred: {err}")
            break

    st.error("Max retries reached. Unable to process the request.")
    return None





# text = """
# Summary from Wikipedia: The article provides an overview of artificial intelligence (AI), its definition, applications, and goals. Here's a summary:

# **Definition and Applications**: AI is intelligence exhibited by machines, particularly computer systems, that enables them to perceive their environment, learn, and take actions to achieve defined goals. Examples of AI applications include web search engines, recommendation systems, speech recognition, autonomous vehicles, and creative tools.

# **Subfields and Goals**: AI research is centered around specific goals, such as reasoning, knowledge representation, planning, learning, and natural language processing. The long-term goal is to achieve general intelligence, equivalent to human-level intelligence.
# """
# text2 ="hello world"
# # Example usage:
# # translated_text = translate_text(text, "en", "fr")
# # print("Translated text:", translated_text)
# # translated_text2 = translate_text(text2, "en", "fr")
# # translated_text2

# # Example usage
# if __name__ == "__main__":
#     try:
#         output = translate_text(
#             text=text,  # The text to translate
#             src_lang="en",  # Source language code (e.g., 'ru' for Russian)
#             tgt_lang="fr"   # Target language code (e.g., 'fr' for French)
#         )
#         # Output the translated text
#         print(output[0]["translation_text"])
#     except ValueError as ve:
#         print(f"ValueError: {ve}")
#     except RuntimeError as re:
#         print(f"RuntimeError: {re}")




# from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv, find_dotenv
# import os

# # Load environment variables
# load_dotenv(find_dotenv())
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# # Define supported languages and their codes
# SUPPORTED_LANGUAGES = {
#     'ar': 'ar_AR', 'cs': 'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX',
#     'et': 'et_EE', 'fi': 'fi_FI', 'fr': 'fr_XX', 'gu': 'gu_IN', 'hi': 'hi_IN',
#     'it': 'it_IT', 'ja': 'ja_XX', 'kk': 'kk_KZ', 'ko': 'ko_KR', 'lt': 'lt_LT',
#     'lv': 'lv_LV', 'my': 'my_MM', 'ne': 'ne_NP', 'nl': 'nl_XX', 'ro': 'ro_RO',
#     'ru': 'ru_RU', 'si': 'si_LK', 'tr': 'tr_TR', 'vi': 'vi_VN', 'zh': 'zh_CN',
#     'af': 'af_ZA', 'az': 'az_AZ', 'bn': 'bn_IN', 'fa': 'fa_IR', 'he': 'he_IL',
#     'hr': 'hr_HR', 'id': 'id_ID', 'ka': 'ka_GE', 'km': 'km_KH', 'mk': 'mk_MK',
#     'ml': 'ml_IN', 'mn': 'mn_MN', 'mr': 'mr_IN', 'pl': 'pl_PL', 'ps': 'ps_AF',
#     'pt': 'pt_XX', 'sv': 'sv_SE', 'sw': 'sw_KE', 'ta': 'ta_IN', 'te': 'te_IN',
#     'th': 'th_TH', 'tl': 'tl_XX', 'uk': 'uk_UA', 'ur': 'ur_PK', 'xh': 'xh_ZA',
#     'gl': 'gl_ES', 'sl': 'sl_SI'
# }

# def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
#     """
#     Translate text using Hugging Face's mbart-large-50-many-to-many-mmt model via the API.
    
#     Args:
#         text (str): The input text to be translated.
#         src_lang (str): The source language code.
#         tgt_lang (str): The target language code.

#     Returns:
#         str: The translated text.
#     """
#     # Validate source and target language codes
#     if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
#         raise ValueError(f"Unsupported language code(s). Supported languages are: {', '.join(SUPPORTED_LANGUAGES.keys())}")

#     # Initialize HuggingFaceEndpoint with max_new_tokens and do_sample directly
#     llm = HuggingFaceEndpoint(
#         endpoint_url="https://api-inference.huggingface.co/models/facebook/mbart-large-50-many-to-many-mmt",
#         huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
#         task="translation",
#         max_new_tokens=512,             # Directly passed
#         do_sample=False,                # Directly passed
#         model_kwargs={
#             "src_lang": SUPPORTED_LANGUAGES[src_lang],  # Passed in model_kwargs
#             "tgt_lang": SUPPORTED_LANGUAGES[tgt_lang]   # Passed in model_kwargs
#         }
#     )

#     # Invoke the model with the text
#     translated_text = llm.invoke(text)

#     return translated_text
   

# # Example usage
# text_to_translate = 'good boy'
# src_language = 'en'
# tgt_language = 'fr'
# translation = translate_text(text_to_translate, src_language, tgt_language)
# print(translation)



# from googletrans import Translator

# def split_text_into_chunks(text: str, chunk_size: int = 5000) -> list:
#     """
#     Splits the text into chunks to manage large text for translation.

#     Args:
#         text (str): The text to split.
#         chunk_size (int): The maximum size of each chunk.

#     Returns:
#         list: A list of text chunks.
#     """
#     chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
#     return chunks

# def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
#     """
#     Translates text from the source language to the target language using googletrans.

#     Args:
#         text (str): The input text to translate.
#         src_lang (str): The source language code (e.g., 'en' for English).
#         tgt_lang (str): The target language code (e.g., 'fr' for French).

#     Returns:
#         str: The translated text.
#     """
#     try:
#         translator = Translator()
#         chunks = split_text_into_chunks(text)

#         translated_chunks = []
#         for chunk in chunks:
#             translated = translator.translate(chunk, src=src_lang, dest=tgt_lang)
#             translated_chunks.append(translated.text)

#         return ' '.join(translated_chunks)
#     except Exception as e:
#         raise RuntimeError(f"An error occurred during translation: {e}")

# # Example usage:
# # translated_text = translate_text("Your long text here", "en", "fr")


# from googletrans import Translator
# def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
#     """
#     Translate text from the source language to the target language using googletrans.

#     Args:
#         text (str): The input text to translate.
#         src_lang (str): The source language code (e.g., 'en' for English).
#         tgt_lang (str): The target language code (e.g., 'fr' for French).

#     Returns:
#         str: The translated text.
#     """
#     try:
#         translator = Translator()
#         translated = translator.translate(text, src=src_lang, dest=tgt_lang)
#         return translated.text
#     except Exception as e:
#         raise RuntimeError(f"An error occurred during translation: {e}")

