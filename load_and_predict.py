import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Define constants
LABELS = ["suicida", "no_suicida"]
OUTPUT_DIR = './trained_model'  # Path to the directory containing the fine-tuned model
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
def classify_single_text(text, model_path=OUTPUT_DIR):
    """
    Classify a single string of text using a fine-tuned BERT model.

    Parameters:
    - text: str, the input text to classify.
    - model_path: str, path to the directory containing the fine-tuned BERT model.

    Returns:
    - predicted_label: str, the predicted label for the input text.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model from the model path
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)  # Move model to GPU

    # Tokenize input text
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    # Move tensors to GPU
    encoding = {key: val.to(device) for key, val in encoding.items()}

    # Set model to evaluation mode and predict
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        pred = torch.argmax(logits, axis=1)
        predicted_label = LABELS[pred.item()]

    return predicted_label

# Example usage
input_text = "Mis sentimientos durante los últimos días. Creo que las personas en Internet son una familia más cercana que las de mi propia casa.Supongo que eso significa que te escribiré esto.Siempre quise ser científico.Desde que era un niño pequeño.Leería, estudiaría y miraría solo para inundar mi imaginación de lo que podía hacer.No puedo ayudar, pero pensar que todas las personas que me ayudaron a tratar de alcanzar ese sueño se sentirán traicionados.Mis padres, mis maestros.todo por nada.Eso es lo que me está desgarrando mientras escribo.El hecho de que lo follé para mí y para todos.El hecho de que otra persona haya conocido a la guadaña.El hecho de que tengo miedo y todo lo que puedo hacer para detenerlo es lo único que lo empeorará para todos los demás.Así que jodidamente cansado de eso.Lamento mis divagaciones y, especialmente, si salen como una perspectiva quejumbrosa sobre las cosas."

predicted_label = classify_single_text(input_text, model_path=OUTPUT_DIR)
print(f"Predicted label: {predicted_label}")
