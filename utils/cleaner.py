import re

def clean_bangla_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs, HTML tags, and non-Bengali characters except spaces
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
    text = " ".join(text.split()) # Multiple spaces remove kora
    return text