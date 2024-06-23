import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRAG:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
        self.vectorizer = TfidfVectorizer().fit(paragraphs)
        self.paragraph_vectors = self.vectorizer.transform(paragraphs)

    def get_most_similar_paragraph(self, question):
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(
            question_vector, self.paragraph_vectors).flatten()
        most_similar_idx = np.argmax(similarities)
        return self.paragraphs[most_similar_idx], similarities[most_similar_idx]

    def answer_question(self, question):
        most_similar_paragraph, similarity = self.get_most_similar_paragraph(
            question)
        return most_similar_paragraph