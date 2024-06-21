import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext

nltk.download('punkt')


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


paragraphs = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.",
    "Deep learning is a subset of machine learning in artificial intelligence that has networks capable of learning unsupervised from data that is unstructured or unlabeled.",
    "Neural networks are a series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.",
    "Supervised learning is a type of machine learning where the model is trained on labeled data.",
    "Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set without pre-existing labels and with a minimum of human supervision.",
    "Reinforcement learning is an area of machine learning where an agent learns to behave in an environment, by performing actions and seeing the results.",
    "Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language.",
    "A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.",
    "Support vector machines (SVM) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis.",
    "K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups).",
    "Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.",
    "Overfitting is a modeling error that occurs when a function is too closely fit to a limited set of data points.",
    "The name of my dog is Snowball."
]


rag_system = SimpleRAG(paragraphs)


class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QnA")

        self.label = tk.Label(root, text="Enter your question:")
        self.label.pack(pady=10)

        self.entry = tk.Entry(root, width=50, justify="center")
        self.entry.pack(pady=10)

        self.button = tk.Button(root, text="Get Answer",
                                command=self.get_answer)
        self.button.pack(pady=10)

        self.answer_label = tk.Label(root, text="Answer:")
        self.answer_label.pack(pady=10)

        self.answer_text = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, width=100, height=10)
        self.answer_text.pack(pady=10)

    def get_answer(self):
        question = self.entry.get()
        answer = rag_system.answer_question(question)
        self.answer_text.delete('1.0', tk.END)
        self.answer_text.insert(tk.INSERT, answer)


root = tk.Tk()
app = RAGApp(root)
root.mainloop()
