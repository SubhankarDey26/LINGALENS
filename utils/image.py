# utils/image.py

import cv2
import pytesseract
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect
import os

# Set your tesseract path (ensure this is correct for deployment)
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', r'D:\Clg Softwares\Tesseract-OCR\tesseract.exe')

# Preprocessing image before OCR
def pre_processing(image_path):
    image = cv2.imread(image_path)
    image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresholding = cv2.adaptiveThreshold(
        image_grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 31, 11
    )
    return adaptive_thresholding

# Custom OCR configuration
custom_config = r'--oem 3 --psm 6'

# Extract text using Tesseract OCR
def text_extraction(image_array):
    text = pytesseract.image_to_string(image_array, lang="eng", config=custom_config)
    return text.strip()

# Mapping detected languages to supported NLLB codes
LANG_MAP = {
    "hi": "hin_Deva",  # Hindi
    "bn": "ben_Beng",  # Bengali
    "or": "ory_Orya",  # Odia
    "en": "eng_Latn"   # English
}

# Load NLLB model (for multiple languages)
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Detect language of input text
def detect_language(text):
    try:
        detected_lang_code = detect(text)
        return LANG_MAP.get(detected_lang_code, "eng_Latn")  # fallback to English
    except Exception as e:
        print(f"Language detection error: {e}")
        return "eng_Latn"

# Translate text from source language to target language
def translate_indictrans(text, src_lang, tgt_lang_code):
    try:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt")

        tgt_lang_token = tokenizer.convert_tokens_to_ids(f"<<{tgt_lang_code}>>")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                forced_bos_token_id=tgt_lang_token
            )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Translation error: {e}")
        return "Translation failed"

# This script is intended to be imported by app.py, not run directly
if __name__ == "__main__":
    print("This module is designed to be imported in app.py")
