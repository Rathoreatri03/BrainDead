from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import os

# Define the directory to save the model
model_directory = "./mental_bart_model"

# Check if the model directory exists, and if not, download and save the model
if not os.path.exists(model_directory):
    # Load the tokenizer and model from the pre-trained version
    tokenizer = BartTokenizer.from_pretrained("Tianlin668/MentalBART")
    model = BartForConditionalGeneration.from_pretrained("Tianlin668/MentalBART")

    # Save the model and tokenizer to the directory
    tokenizer.save_pretrained(model_directory)
    model.save_pretrained(model_directory)
else:
    # Load the tokenizer and model from the saved directory
    tokenizer = BartTokenizer.from_pretrained(model_directory)
    model = BartForConditionalGeneration.from_pretrained(model_directory)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to generate text
def generate_response(input_text):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)

    # Move inputs to GPU if available
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate text using the model
    outputs = model.generate(**inputs, max_length=1000, num_beams=5, early_stopping=True)

    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Example usage
if __name__ == "__main__":
    input_text = "I'm so tired of everything. It's like nothing will ever get better, and I'm just stuck in this cycle. Nothing excites me, and I don't really see the point in doing anything anymore. I don't even know why I'm still here sometimes."
    response = generate_response(input_text)

    print("Input Text:", input_text)
    print("Generated Response:", response)
