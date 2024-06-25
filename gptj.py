from transformers import GPTNeoForCausalLM, GPT2Tokenizer

class GPTJTextGenerator:
    def __init__(self):
        self.model_name = "EleutherAI/gpt-neo-2.7B"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name, padding_side='left')
        self.model = GPTNeoForCausalLM.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_text(self, prompt):
        print("PROMPT: ", prompt)
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Stop sequence
        output_ids = self.model.generate(input_ids, attention_mask=attention_mask, max_length=150, eos_token_id=self.tokenizer.eos_token_id)
        
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("BEFORE: ", answer)
        answer = answer.split("QUESTION:")[0].strip()
        print("AFTER: ", answer)
        return answer