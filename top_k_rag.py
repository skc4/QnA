import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import hashlib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

class TopKRAG:
    def __init__(self, paragraphs, stop_words='english', ngram_range=(1, 2), cache_size=100):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(stop_words)) if stop_words == 'english' else set()
        self.paragraphs = [self.preprocess_text(paragraph) for paragraph in paragraphs]
        self.vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range).fit(self.paragraphs)
        self.paragraph_vectors = self.vectorizer.transform(self.paragraphs)
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
        
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.paragraph_vectors).flatten()
        top_k_indices = similarities.argsort()[-k:][::-1]
        top_k_paragraphs = [self.paragraphs[idx] for idx in top_k_indices]
        
        # Cache
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))  # Remove oldest
        self.cache[cache_key] = (top_k_paragraphs, similarities[top_k_indices])
        
        logging.info(f"Computed top {k} paragraphs for question: {question}")
        return top_k_paragraphs, similarities[top_k_indices]

    def answer_question(self, question, k=3):
        try:
            top_k_paragraphs, similarities = self.get_top_k_paragraphs(question, k)
            return top_k_paragraphs
        except Exception as e:
            logging.error(f"Error answering question: {question} - {e}")
            return ["An error occurred while processing your question."]