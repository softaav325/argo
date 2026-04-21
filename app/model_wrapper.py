import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentModel:
    def __init__(self, model_path: str = "./model_artifacts"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        # Определяем устройство (CPU для этого демо)
        self.device = torch.device("cpu")
        
    def load(self):
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        logger.info("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded.")

    def predict(self, text: str) -> str:
        if self.model is None:
            self.load()

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pos_score = probs[0][1].item()
        neg_score = probs[0][0].item()

        if pos_score > neg_score:
            return f"😊 Позитив ({pos_score:.2%})"
        else:
            return f"😠 Негатив ({neg_score:.2%})"

model_wrapper = SentimentModel()