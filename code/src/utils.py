import re
import unicodedata
from pyvi import ViTokenizer

def normalize_text(text):
    """
    Normalize text by converting to lowercase, removing special characters,
    and normalizing unicode.
    """
    if not isinstance(text, str):
        return ""
    
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

def segment_text(text):
    """
    Segment text using pyvi.
    PhoBERT requires word-segmented input (e.g., "máy tính" -> "máy_tính").
    """
    return ViTokenizer.tokenize(text)

def preprocess_text(text):
    """
    Full preprocessing pipeline.
    """
    text = normalize_text(text)
    text = segment_text(text)
    return text
