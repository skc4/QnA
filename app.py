import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog
import os
import shutil
import threading
import start_db
import contextlib
import io
import warnings


warnings.filterwarnings("ignore", category=UserWarning, message=".*LangChainDeprecationWarning.*")

class RAGApp:
    def __init__(self, root, get_detailed_answer, filename, upload_directory):
        self.get_detailed_answer = get_detailed_answer
        self.root = root
        self.upload_directory = upload_directory
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


        self.upload_button = tk.Button(root, text="Clear DB", command=self.clear_db)
        self.upload_button.pack(pady=10)


        self.upload_button = tk.Button(root, text="Upload File", command=self.upload_file)
        self.upload_button.pack(pady=10)

        

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

    def clear_db(self):
        start_db.clear_database()
        self.answer_text.insert(tk.INSERT, f"\nDatabase Cleared")

    def run_spin_db(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            start_db.spin_db()
        self.answer_text.insert(tk.INSERT, output.getvalue())

    def upload_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                destination = os.path.join(self.upload_directory, os.path.basename(file_path))
                shutil.copy(file_path, destination)
                self.filename_label.config(text=f"Uploaded file: {os.path.basename(file_path)}")
                self.answer_text.insert(tk.INSERT, f"\nFile uploaded to: {destination}")
                self.run_spin_db()
            except Exception as e:
                self.answer_text.insert(tk.INSERT, f"\nError uploading file: {str(e)}")


if __name__ == "__main__":
    upload_dir = "./data"
    os.makedirs(upload_dir, exist_ok=True)
    
    root = tk.Tk()
    app = RAGApp(root, lambda question: "This is a test.", __file__, upload_dir)
    root.mainloop()