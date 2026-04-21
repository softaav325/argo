# ArgoCD

[ArgoCD GitHub](https://github.com/argoproj/argo-cd)

[Getting Started](https://argo-cd.readthedocs.io/en/stable/getting_started/)

[Multiple configuration objects](https://argo-cd.readthedocs.io/en/stable/operator-manual/declarative-setup/#multiple-configuration-objects)



demo/
├── app/
│   ├── train.py          # Скрипт тренировки
│   ├── model_wrapper.py  # Обертка для загрузки модели
│   └── gradio_app.py     # Веб-интерфейс
├── Dockerfile
├── requirements.txt
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml      # Опционально, если есть Ingress Controller
└── README.md



app/train.py
Этот скрипт тренирует модель на небольшом датасете. Для скорости используем синтетические данные или маленький subset.

app/gradio_app.py
Это веб-интерфейс. Он загружает обученную модель и предоставляет интерфейс для ввода текста.