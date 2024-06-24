import nltk
from enhanced_vector_rag import EnhancedVectorRAG
from gptj import GPTJTextGenerator
from app import RAGApp
import tkinter as tk

nltk.download('punkt')

paragraphs = [
    "The Amazon rainforest, often referred to as the “lungs of the Earth,” covers approximately 5.5 million square kilometers.",
    "It is home to an incredibly diverse range of species, including many that are yet to be discovered.",
    "The rainforest plays a critical role in regulating the global climate by absorbing carbon dioxide and releasing oxygen.",
    "However, deforestation poses a significant threat to this vital ecosystem, driven primarily by logging, agriculture, and mining activities.",
    "Efforts to combat deforestation include reforestation projects and stricter enforcement of environmental laws.",
    "Indigenous communities in the Amazon play a crucial role in preserving the rainforest through sustainable practices.",
    "Climate change, exacerbated by deforestation, is leading to changes in rainfall patterns and more frequent extreme weather events.",
    "Protecting the Amazon is essential not only for biodiversity but also for the overall health of our planet.",
    "Artificial Intelligence (AI) has seen tremendous growth over the past decade, revolutionizing various industries from healthcare to finance.",
    "Machine learning, a subset of AI, involves training algorithms on large datasets to make predictions or decisions without explicit programming.",
    "Recent advancements in neural networks, particularly deep learning, have enabled significant improvements in image and speech recognition, natural language processing, and autonomous systems.",
    "AI-powered tools are being used to analyze medical images, helping doctors diagnose diseases more accurately and quickly.",
    "In finance, AI algorithms are employed to detect fraudulent transactions and predict market trends.",
    "The ethical implications of AI, including issues of bias and privacy, are subjects of ongoing debate.",
    "Continued research and development in AI aim to create more generalizable and robust systems that can operate in a variety of real-world environments."
]

rag_system = EnhancedVectorRAG(paragraphs)
text_generator = GPTJTextGenerator()

def validate_input(question):
    if not isinstance(question, str) or len(question) == 0:
        raise ValueError("Question must be a non-empty string")
    return question

def get_detailed_answer(question):
    question = validate_input(question)
    top_k_paragraphs = rag_system.answer_question(question, k=3)
    combined_paragraphs = " ".join(top_k_paragraphs)
    prompt = f"DOCUMENT: {combined_paragraphs} \nQUESTION:{question} \nINSTRUCTIONS: Answer the QUESTION using the DOCUMENT text above. Keep your answer ground in the facts of the DOCUMENT. If the DOCUMENT does not contain the facts to answer the QUESTION return NONE"
    detailed_answer = text_generator.generate_text(prompt)
    return detailed_answer

root = tk.Tk()
app = RAGApp(root, get_detailed_answer, __file__)
root.mainloop()

# questions = [
#     "How does the Amazon rainforest contribute to regulating the global climate, and what are the primary threats to this ecosystem?"
# ]

# for question in questions:
#     detailed_answer = get_detailed_answer(question)
#     print(f"\nQuestion: {question}")
#     print(f"\nDetailed Answer: {detailed_answer}\n")