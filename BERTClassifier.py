from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch.nn as nn
import torch


class JapaneseBERTClassfier:
    def __init__(self, model_path, tokenizer_path, device, num_labels=32):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        print('num_labels', num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text):
        inputs = self.tokenizer(text, add_special_tokens=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        ps = nn.Softmax(1)(outputs.logits)

        max_p = torch.max(ps)
        result = torch.argmax(ps).item() if max_p > 0.8 else -1
        return result