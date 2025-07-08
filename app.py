import pickle
import gradio as gr
import pandas as pd

# Load your model and preprocessor
with open("/Users/MU20414673/Krish_Naik_Project/mlproject/artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("/Users/MU20414673/Krish_Naik_Project/mlproject/artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

def predict_fn(*inputs):
    # Define the column names in the same order as input_components
    columns = [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
        "reading_score",
        "writing_score"
    ]
    # Convert inputs to DataFrame
    data = pd.DataFrame([inputs], columns=columns)
    processed = preprocessor.transform(data)
    prediction = model.predict(processed)
    return str(prediction[0])

input_components = [
    gr.Textbox(label="Gender"),
    gr.Textbox(label="Race Ethnicity"),
    gr.Textbox(label="Parental Level of Education"),
    gr.Textbox(label="Lunch"),
    gr.Textbox(label="Test Preparation Course"),
    gr.Textbox(label="Reading Score"),
    gr.Textbox(label="Writing Score"),
]


iface = gr.Interface(
    fn=predict_fn,
    inputs=input_components,
    outputs="text",
    title="ML Model Predictor",
    description="Enter feature values to get predictions."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8080)