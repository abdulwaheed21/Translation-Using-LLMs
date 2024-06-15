from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from bs4 import BeautifulSoup
import contractions


# Load models directly
en_tokenizer = AutoTokenizer.from_pretrained("abdulwaheed1/english-to-urdu-translation-mbart")
en_model = AutoModelForSeq2SeqLM.from_pretrained("abdulwaheed1/english-to-urdu-translation-mbart")

ur_tokenizer = AutoTokenizer.from_pretrained("abdulwaheed1/urdu_to_english_translation_mbart")
ur_model = AutoModelForSeq2SeqLM.from_pretrained("abdulwaheed1/urdu_to_english_translation_mbart")

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove symbols
    text = re.sub(r'[^\w\s]', '', text)
    
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove extra spaces and normalize spaces between words
    #text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def translate_text(text, tokenizer, model):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)

    # Generate translation
    translation_ids = model.generate(**inputs)
    translation = tokenizer.batch_decode(translation_ids, skip_special_tokens=True)[0]
    
    return translation
