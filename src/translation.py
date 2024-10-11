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
    'arabic': 'ar_AR', 'czech': 'cs_CZ', 'german': 'de_DE', 'english': 'en_XX', 'spanish': 'es_XX',
    'estonian': 'et_EE', 'finnish': 'fi_FI', 'french': 'fr_XX', 'gujarati': 'gu_IN', 'hindi': 'hi_IN',
    'italian': 'it_IT', 'japanese': 'ja_XX', 'kazakh': 'kk_KZ', 'korean': 'ko_KR', 'lithuanian': 'lt_LT',
    'latvian': 'lv_LV', 'burmese': 'my_MM', 'nepali': 'ne_NP', 'dutch': 'nl_XX', 'romanian': 'ro_RO',
    'russian': 'ru_RU', 'sinhala': 'si_LK', 'turkish': 'tr_TR', 'vietnamese': 'vi_VN', 'chinese': 'zh_CN',
    'afrikaans': 'af_ZA', 'azerbaijani': 'az_AZ', 'bengali': 'bn_IN', 'persian': 'fa_IR', 'hebrew': 'he_IL',
    'croatian': 'hr_HR', 'indonesian': 'id_ID', 'georgian': 'ka_GE', 'khmer': 'km_KH', 'macedonian': 'mk_MK',
    'malayalam': 'ml_IN', 'mongolian': 'mn_MN', 'marathi': 'mr_IN', 'polish': 'pl_PL', 'pashto': 'ps_AF',
    'portuguese': 'pt_XX', 'swedish': 'sv_SE', 'swahili': 'sw_KE', 'tamil': 'ta_IN', 'telugu': 'te_IN',
    'thai': 'th_TH', 'tagalog': 'tl_XX', 'ukrainian': 'uk_UA', 'urdu': 'ur_PK', 'xhosa': 'xh_ZA',
    'galician': 'gl_ES', 'slovenian': 'sl_SI'
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

