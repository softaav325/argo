FROM pytorch/pytorch:2.0-cuda11.7-runtime

WORKDIR /app
COPY src/train.py .

# Создаём директорию для вывода
RUN mkdir -p /output

CMD ["python", "train.py"]
