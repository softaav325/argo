import torch
import torch.nn as nn
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader

# --- КОНФИГУРАЦИЯ ---
SEQ_LENGTH = 50
BATCH_SIZE = 32 # уменьшаем с 128
EPOCHS = 15
EMBED_SIZE = 64
HIDDEN_SIZE = 128 # 256 Чуть больше для лучшего качества
NUM_LAYERS = 2
LEARNING_RATE = 0.002
DATA_FILE = "./yesenin.txt" # Файл должен лежать рядом со скриптом при запуске

class CharDataset(Dataset):
    def __init__(self, text, seq_length=SEQ_LENGTH):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Кодирование всего текста в индексы
        self.encoded_data = [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
        self.seq_length = seq_length
        
    def __len__(self):
        return max(0, len(self.encoded_data) - self.seq_length)

    def __getitem__(self, idx):
        x = self.encoded_data[idx:idx + self.seq_length]
        y = self.encoded_data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class TextGeneratorModel(nn.Module):
    def __init__(self, vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        output = self.fc(output)
        return output, hidden

def train_model():
    print(f">>> Загрузка данных из {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        # Фоллбэк на короткие данные, если файла нет (для тестов)
        print("Файл не найден, использую демо-текст.")
        text = "В лесу родилась елочка..." * 100
    else:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            text = f.read()
    
    # Очистка текста (опционально): оставляем буквы, пробелы и знаки препинания
    # Можно убрать лишние переносы строк, заменив их на пробелы
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    print(f">>> Размер текста: {len(text)} символов.")
    
    dataset = CharDataset(text)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Сохраняем метаданные
    meta_data = {
        "char_to_idx": dataset.char_to_idx,
        "idx_to_char": dataset.idx_to_char,
        "vocab_size": dataset.vocab_size
    }
    
    model = TextGeneratorModel(dataset.vocab_size)
    device = torch.device("cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(">>> Начало тренировки...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            
            loss = criterion(outputs.reshape(-1, dataset.vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 3 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # Сохранение
    save_path = "./model_artifacts"
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    with open(os.path.join(save_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False)
        
    print(f">>> Модель сохранена в {save_path}")

if __name__ == "__main__":
    train_model()