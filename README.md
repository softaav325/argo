# ArgoCD

[ArgoCD GitHub](https://github.com/argoproj/argo-cd)

[Getting Started](https://argo-cd.readthedocs.io/en/stable/getting_started/)

[Multiple configuration objects](https://argo-cd.readthedocs.io/en/stable/operator-manual/declarative-setup/#multiple-configuration-objects)

Структура репозитория:

AI-emotion/
├── app/
│   ├── train.py          # Скрипт тренировки
│   ├── model_wrapper.py  # Обертка для загрузки модели
│   └── gradio_app.py     # Веб-интерфейс
├── Dockerfile
├── requirements-build.txt
├── requirements-run.t
└── README.md

AI-koder/
├── app/
│   ├── train.py          # Скрипт тренировки
│   ├── model_wrapper.py  # Обертка для загрузки модели
│   └── gradio_app.py     # Веб-интерфейс
├── Dockerfile
├── requirements-build.txt
├── requirements-run.t
└── README.md


AI-emotion/
├── app/
│   ├── train.py          # Скрипт тренировки
│   ├── model_wrapper.py  # Обертка для загрузки модели
│   └── gradio_app.py     # Веб-интерфейс
├── Dockerfile
├── requirements-build.txt
├── requirements-run.t
└── README.md



Локальная сборка образов: 
``` git clone https://github.com/softaav325/argo.git ```

Переход в нужную папку для сборки
``` cd <dir> ```    

Сборка
``` docker build -t demo:latest .```