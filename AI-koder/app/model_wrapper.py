import torch
import torch.nn as nn
import json
import os
from train import TextGeneratorModel # Импортируем класс модели из train.py

class GeneratorWrapper:
    def __init__(self, model_path="./model_artifacts"):
        self.model_path = model_path
        self.model = None
        self.idx_to_char = {}
        self.char_to_idx = {}
        self.device = torch.device("cpu")

    def load(self):
        if self.model is not None:
            return
        
        with open(os.path.join(self.model_path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        self.idx_to_char = {int(k): v for k, v in meta["idx_to_char"].items()}
        self.char_to_idx = meta["char_to_idx"]
        vocab_size = meta["vocab_size"]
        
        self.model = TextGeneratorModel(vocab_size)
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, "model.pth"), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def generate(self, seed_text, length=200, temperature=0.8):
        self.load()
        
        # Подготовка входных данных
        input_seq = [self.char_to_idx.get(ch, 0) for ch in seed_text]
        
        current_input = torch.tensor([input_seq[-50:]], dtype=torch.long).to(self.device)
        generated = list(seed_text)
        
        hidden = None
        
        with torch.no_grad():
            for _ in range(length):
                output, hidden = self.model(current_input, hidden)
                
                last_output = output[0, -1, :] / temperature
                probs = torch.softmax(last_output, dim=0)
                
                next_idx = torch.multinomial(probs, 1).item()
                next_char = self.idx_to_char.get(next_idx, "")
                
                generated.append(next_char)
                
                current_input = torch.tensor([[next_idx]], dtype=torch.long).to(self.device)
                
        return "".join(generated)

# Создаем глобальный экземпляр
generator = GeneratorWrapper()