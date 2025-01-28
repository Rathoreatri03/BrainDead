# Import required libraries
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import matplotlib.pyplot as plt

# Define model and tokenizer names
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_dir = "./sentiment_model"

# Check if the model directory exists
if not os.path.exists(model_dir):
    print("Downloading the model...")
    # Download the model and tokenizer locally
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
else:
    print("Loading model from local directory...")
    # Load the model and tokenizer from local directory
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Check if GPU is available and set device
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load the sentiment analysis pipeline with the specified device
sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=device)

# Input texts from the user
texts = [
    "I love the new design of this product! It's so sleek and user-friendly.",
    "This is the worst experience I've ever had with customer service.",
    "I'm feeling neutral about this decision.",
    "The movie was absolutely amazing, I cried and laughed so much!",
    "This weather is making me feel very gloomy and unmotivated."
]

# Analyze sentiments
results = sentiment_pipeline(texts)

# Extract labels and scores
labels = [result['label'] for result in results]
scores = [result['score'] for result in results]

# Count sentiment types
sentiment_counts = {label: labels.count(label) for label in set(labels)}

# Display results and analyze mood/personality
print("Sentiment Analysis Results:")
for i, text in enumerate(texts):
    print(f"Text: {text}")
    print(f"Sentiment: {results[i]['label']}, Confidence: {results[i]['score']:.2f}")
    print()

# Mood/Personality Analysis (simple heuristic)
mood_map = {
    'POSITIVE': 'optimistic, cheerful',
    'NEGATIVE': 'pessimistic, stressed',
    'NEUTRAL': 'balanced, thoughtful'
}

overall_mood = mood_map[max(sentiment_counts, key=sentiment_counts.get)]
print(f"Overall Mood/Personality Inference: {overall_mood}")

# Create a pie chart for sentiment distribution
plt.figure(figsize=(8, 6))
plt.pie(
    sentiment_counts.values(),
    labels=sentiment_counts.keys(),
    autopct='%1.1f%%',
    startangle=140,
    colors=['#8bc34a', '#e57373', '#ffc107']  # Green for positive, red for negative, yellow for neutral
)
plt.title('Sentiment Distribution')
plt.show()
