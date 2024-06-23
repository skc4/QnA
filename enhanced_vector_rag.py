import numpy as np
from sentence_transformers import SentenceTransformer, util
import hashlib
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')


# Sentence Transformers for more accurate vector-based retrieval
class EnhancedVectorRAG:
    def __init__(self, paragraphs, model_name='paraphrase-MiniLM-L6-v2', cache_size=100):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.paragraphs = [self.preprocess_text(paragraph) for paragraph in paragraphs]
        self.model = SentenceTransformer(model_name)
        self.paragraph_vectors = self.model.encode(self.paragraphs, convert_to_tensor=True)
        self.cache = {}
        self.cache_size = cache_size
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        text = ' '.join(words)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_top_k_paragraphs(self, question, k=3):
        question = self.preprocess_text(question)
        
        # Cache
        cache_key = hashlib.md5(question.encode('utf-8')).hexdigest()
        if cache_key in self.cache:
            logging.info(f"Cache hit for question: {question}")
            return self.cache[cache_key]
        
        question_vector = self.model.encode(question, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(question_vector, self.paragraph_vectors)[0]
        top_k_indices = similarities.argsort(descending=True)[:k]
        top_k_paragraphs = [self.paragraphs[idx] for idx in top_k_indices]
        
        # Cache
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))  # Remove oldest
        self.cache[cache_key] = (top_k_paragraphs, similarities[top_k_indices].cpu().numpy())
        
        logging.info(f"Computed top {k} paragraphs for question: {question}")
        return top_k_paragraphs, similarities[top_k_indices].cpu().numpy()

    def answer_question(self, question, k=3):
        try:
            top_k_paragraphs, similarities = self.get_top_k_paragraphs(question, k)
            return top_k_paragraphs
        except Exception as e:
            logging.error(f"Error answering question: {question} - {e}")
            return ["An error occurred while processing your question."]