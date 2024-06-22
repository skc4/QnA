from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GPTJTextGenerator:
    def __init__(self, model_name='EleutherAI/gpt-j-6B'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
    def generate_text(self, prompt, max_new_tokens=150):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        with torch.no_grad():  # Disable gradient computation
            outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text