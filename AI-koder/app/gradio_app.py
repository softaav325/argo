import gradio as gr
from model_wrapper import generator

def generate_text(seed, length, temperature):
    try:
        result = generator.generate(seed, int(length), float(temperature))
        return result
    except Exception as e:
        return f"Ошибка: {str(e)}"

iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Начни фразу (например: 'Белеет парус...')", label="Затравка"),
        gr.Slider(10, 500, value=100, step=10, label="Длина текста"),
        gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Температура (креативность)")
    ],
    outputs=gr.Textbox(label="Сгенерированный текст"),
    title="🧠 AI Генератор Стихов (Есенин Style)",
    description="Модель LSTM, обученная на стихах. Введи начало строки."
)

if __name__ == "__main__":
    print(">>> Загрузка модели в память...")
    generator.load()
    print(">>> Модель готова. Запуск веб-сервера...")
    iface.launch(server_name="0.0.0.0", server_port=7860)