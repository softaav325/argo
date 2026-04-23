AI-koder/
├── app/
│   ├── train.py
│   ├── model_wrapper.py
│   └── gradio_app.py
├── Dockerfile
├── requirements-build.txt
└── requirements-run.txt

# Генеративную модель (Text Generation). Тренируем маленькую нейросеть  писать код или стихи в стиле конкретного автора. Вводится начало фразы, а модель дописывает её в заданном стиле.

# Generative AI. Работа с последовательностями: Показывает понимание RNN/LSTM/Transformers. Легковесность: Модель будет весить пару мегабайт, тренировка займет 5-10 минут на CPU.

Сборка локально ```docker build -t demo:latest .```