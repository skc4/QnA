import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TopKRAG:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit(paragraphs)
        self.paragraph_vectors = self.vectorizer.transform(paragraphs)
    
    def get_top_k_paragraphs(self, question, k=3):
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.paragraph_vectors).flatten()
        top_k_indices = similarities.argsort()[-k:][::-1]
        top_k_paragraphs = [self.paragraphs[idx] for idx in top_k_indices]
        return top_k_paragraphs, similarities[top_k_indices]

    def answer_question(self, question, k=3):
        top_k_paragraphs, similarities = self.get_top_k_paragraphs(question, k)
        return top_k_paragraphs