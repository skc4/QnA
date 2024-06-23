import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
import os
import threading

class RAGApp:
    def __init__(self, root, get_detailed_answer, filename):
        self.get_detailed_answer = get_detailed_answer
        self.root = root
        self.root.title("QnA")

        self.label = tk.Label(root, text="Enter your question:")
        self.label.pack(pady=10)

        self.entry = tk.Entry(root, width=50, justify="center")
        self.entry.pack(pady=10)

        self.button = tk.Button(root, text="Get Answer", command=self.display_answer)
        self.button.pack(pady=10)

        self.answer_label = tk.Label(root, text="Answer:")
        self.answer_label.pack(pady=10)

        self.answer_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=10)
        self.answer_text.pack(pady=10)

        self.filename_label = tk.Label(root, text=f"Running from file: {os.path.basename(filename)}")
        self.filename_label.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        self.progress.pack(pady=10)

    def display_answer(self):
        question = self.entry.get()
        self.progress.start()
        self.answer_text.delete('1.0', tk.END)

        def generate_answer():
            answer = self.get_detailed_answer(question)
            self.root.after(0, self.update_answer_text, answer)

        threading.Thread(target=generate_answer).start()

    def update_answer_text(self, answer):
        self.progress.stop()
        self.answer_text.insert(tk.INSERT, answer)