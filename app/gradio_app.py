import gradio as gr
from model_wrapper import model_wrapper

def predict(text):
    try:
        return model_wrapper.predict(text)
    except Exception as e:
        return f"Error: {e}"

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Введите текст..."),
    outputs=gr.Textbox(label="Результат"),
    title="Sentiment Analysis Demo (PyTorch)",
    examples=[["Отличный продукт!"], ["Ужасный сервис."]]
)

if __name__ == "__main__":
    model_wrapper.load() # Pre-load
    iface.launch(server_name="0.0.0.0", server_port=7860)