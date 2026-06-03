import gradio as gr #type: ignore
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('Gold.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def calculate_gold_rate(usd_inr):
    try:
        scaled_input = scaler.transform(
            np.array([[float(usd_inr)]])
        )

        prediction = model.predict(scaled_input)

        return round(float(np.ravel(prediction)[0]), 2)

    except Exception as e:
        return str(e)

demo = gr.Interface(
    fn=calculate_gold_rate,
    inputs=gr.Number(label="USD_INR"),
    outputs=gr.Textbox(label="Predicted Gold Rate"),
    title="How Much is 1g Gold Now in INR"
)

demo.launch(share=True)