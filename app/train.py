import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score
import os

# Используем легкую модель для демо
MODEL_NAME = "cointegrated/rubert-tiny2" 

def prepare_dataset():
    # Синтетические данные
    texts = [
        "Отличный сервис, очень доволен!", "Ужасное качество, не рекомендую.",
        "Быстрая доставка, спасибо.", "Посылка пришла разбитой, кошмар.",
        "Хороший товар за свои деньги.", "Деньги на ветер, больше не куплю.",
        "Рекомендую всем друзьям.", "Служба поддержки игнорирует запросы.",
        "Превзошло мои ожидания.", "Полный развод, берегитесь."
    ] * 20
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 20
    
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    return dataset

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train_and_save():
    print(">>> Загрузка токенизатора и модели...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    print(">>> Подготовка данных...")
    dataset = prepare_dataset()
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_strategy="no", # Отключаем логи для чистоты вывода в Docker
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        compute_metrics=compute_metrics,
    )

    print(">>> Начало тренировки...")
    trainer.train()

    # Сохраняем финальную модель
    model_path = "./model_artifacts"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f">>> Модель успешно сохранена в {model_path}")

if __name__ == "__main__":
    train_and_save()