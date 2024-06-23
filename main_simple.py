from rag import SimpleRAG
from app import RAGApp
import tkinter as tk
import nltk
import sys

nltk.download('punkt')


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

def get_answer(question):
    return rag_system.answer_question(question)

root = tk.Tk()
app = RAGApp(root, get_answer, __file__)
root.mainloop()