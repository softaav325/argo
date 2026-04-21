from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

app = Flask(__name__)

# Модель
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Загружаем обученную модель
try:
    model = SimpleNN()
    model.load_state_dict(torch.load('/models/mnist_model.pth', map_location='cpu'))
    model.eval()
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Модель не загружена'}), 500

    try:
        data = request.json
        # Преобразуем массив в тензор
        image_array = np.array(data['image']).reshape(28, 28).astype(np.float32)
        image_tensor = transform(image_array).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, 1).item()
            confidence = probabilities[0][prediction].item()

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def index():
    return '''
    <h1>Распознавание цифр MNIST</h1>
    <p>Нарисуйте цифру в поле ниже и нажмите "Распознать"</p>
    <canvas id="canvas" width="280" height="280" style="border:1px solid #000;"></canvas>
    <br>
    <button onclick="predict()">Распознать</button>
    <div id="result"></div>

    <script>
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, 280, 280);

      let isDrawing = false;
      canvas.addEventListener('mousedown', () => isDrawing = true);
      canvas.addEventListener('mouseup', () => isDrawing = false);
      canvas.addEventListener('mousemove', (e) => {
        if (isDrawing) {
          ctx.fillStyle = 'black';
          ctx.beginPath();
          ctx.arc(e.offsetX, e.offsetY, 10, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      async function predict() {
        const imageData = ctx.getImageData(0, 0, 280, 280).data;
        const pixels = [];

        // Извлекаем яркость каждого пикселя (усредняем RGB)
        for (let i = 0; i < imageData.length; i += 4) {
          const r = imageData[i];
          const g = imageData[i + 1];
          const b = imageData[i + 2];
          // Инвертируем: белый фон (255) → 0, чёрный (0) → 255
          const brightness = 255 - Math.round((r + g + b) / 3);
          pixels.push(brightness);
        }

        // Уменьшаем изображение с 280×280 до 28×28 методом усреднения
        const resized = new Array(784).fill(0); // 28*28
        for (let y = 0; y < 28; y++) {
          for (let x = 0; x < 28; x++) {
            let sum = 0;
            for (let dy = 0; dy < 10; dy++) {
              for (let dx = 0; dx < 10; dx++) {
                const srcX = x * 10 + dx;
                const srcY = y * 10 + dy;
                sum += pixels[srcY * 280 + srcX];
              }
            }
            resized[y * 28 + x] = sum / 100; // Среднее значение
          }
        }

        try {
          const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image: resized})
          });
          const result = await response.json();

          if (result.error) {
            document.getElementById('result').innerHTML =
              `<p style="color: red;">Ошибка: ${result.error}</p>`;
          } else {
            document.getElementById('result').innerHTML =
              `<h2>Модель предсказала: <strong>${result.prediction}</strong></h2>
               <p>Уверенность: <strong>${(result.confidence * 100):.2f}%</strong></p>`;
          }
        } catch (error) {
          document.getElementById('result').innerHTML =
            `<p style="color: red;">Ошибка связи с сервером</p>`;
        }
      }
    </script>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
