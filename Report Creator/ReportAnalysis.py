from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, BartTokenizer, BartForConditionalGeneration
import torch
import os
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO

# Define model and tokenizer for Sentiment Analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_dir = "./sentiment_model"

# Check if model is already downloaded
if not os.path.exists(model_dir):
    print("Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
else:
    print("Loading model from local directory...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Set device (GPU/CPU)
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=device)

# Sample Text Inputs
texts = [
    "I love the new design of this product! It's so sleek and user-friendly.",
    "This is the worst experience I've ever had with customer service.",
    "I'm feeling neutral about this decision.",
    "The movie was absolutely amazing, I cried and laughed so much!",
    "This weather is making me feel very gloomy and unmotivated."
]

# Analyze Sentiments
results = sentiment_pipeline(texts)

# Extract Labels
labels = [result['label'] for result in results]

# Count Sentiment Types
sentiment_counts = {label: labels.count(label) for label in set(labels)}

# Generate Larger Pie Chart
plt.figure(figsize=(5, 5))
plt.pie(
    sentiment_counts.values(),
    labels=sentiment_counts.keys(),
    autopct='%1.1f%%',
    startangle=140,
    colors=['#8bc34a', '#e57373', '#ffc107']
)
plt.title('Sentiment Distribution', fontsize=14)
chart_buffer = BytesIO()
plt.savefig(chart_buffer, format='png')
chart_buffer.seek(0)
plt.close()

# Mood/Personality Mapping
mood_map = {
    'POSITIVE': 'optimistic, cheerful',
    'NEGATIVE': 'pessimistic, stressed',
    'NEUTRAL': 'balanced, thoughtful'
}
overall_mood = mood_map[max(sentiment_counts, key=sentiment_counts.get)]

# Load Banner Image
banner_path = "2.jpg"

# Generate PDF with Narrow Margins
pdf_path = "Sentiment_Analysis_Report.pdf"
margin = 0.5 * inch  # Reduced margins

doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    leftMargin=margin,
    rightMargin=margin,
    topMargin=margin,
    bottomMargin=margin
)

styles = getSampleStyleSheet()

# Custom Styles
styles.add(ParagraphStyle(name='TitleStyle', fontSize=22, textColor=colors.darkblue, spaceAfter=20))
styles.add(ParagraphStyle(name='HeadingStyle', fontSize=16, textColor=colors.darkred, spaceAfter=15))
styles.add(ParagraphStyle(name='NormalStyle', fontSize=12, leading=15, spaceAfter=10))
styles.add(ParagraphStyle(name='HighlightStyle', fontSize=14, textColor=colors.white, backColor=colors.red, bold=True, spaceAfter=20))

# Story Content
story = []

# Add Banner (Full-Width)
banner = Image(banner_path, width=7.5 * inch, height=3 * inch)  # Adjusted for narrow margins
story.append(banner)
story.append(Spacer(1, 20))

# Add Title
story.append(Paragraph("Summary of Sentiment Analysis Results", styles['HeadingStyle']))
story.append(Spacer(1, 10))

# Table Data for Sentiment Analysis (No Confidence Column)
summary_data = [["Text", "Sentiment"]]
for i, text in enumerate(texts):
    summary_data.append([
        Paragraph(text, styles["NormalStyle"]),  # Auto-wrap text
        results[i]['label']
    ])

# Table Styling for Sentiment Analysis Results
summary_table = Table(summary_data, colWidths=[5.5 * inch, 2 * inch])  # Adjusted width for new margins
summary_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkred),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
story.append(summary_table)
story.append(Spacer(1, 40))


# Add Title for Overall Mood
story.append(Paragraph("Overall Mood/Personality Inference", styles['HeadingStyle']))
story.append(Spacer(1, 10))

# Table Data for Overall Mood
mood_data = [["Aspect", "Inference"]]
mood_data.append([
    Paragraph("Overall Mood/Personality Inference", styles["NormalStyle"]),
    f"{overall_mood}"
])

# Table Styling for Overall Mood Inference
mood_table = Table(mood_data, colWidths=[5.5 * inch, 2 * inch])  # Adjusted width for new margins
mood_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkred),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
story.append(mood_table)

# Add Page Break after Sentiment Distribution Chart
story.append(PageBreak())



# Add Sentiment Distribution Chart as before
story.append(Paragraph("Sentiment Distribution Chart", styles['HeadingStyle']))
story.append(Spacer(1, 10))

# Add Large Pie Chart
chart_image = Image(chart_buffer, width=6 * inch, height=6 * inch)  # Adjusted size for narrow margins
story.append(chart_image)

# Add Page Break after Pie Chart and before Final Analysis
story.append(PageBreak())
# Add History Analysis Report Title
story.append(Paragraph("History Analysis Report", styles['HeadingStyle']))
story.append(Spacer(1, 20))
# Add History Analysis Image (a.jpg)
history_image_path = "powerbi.jpg"
history_image = Image(history_image_path, width=6 * inch, height=3.5 * inch)  # Adjusted dimensions
story.append(history_image)
story.append(Spacer(1, 20))

# Add Title for Final Analysis
story.append(Paragraph("Final Analysis of Mental State", styles['HeadingStyle']))
story.append(Spacer(1, 10))

# MentalBART Integration
mental_bart_dir = "./mental_bart_model"
if not os.path.exists(mental_bart_dir):
    bart_tokenizer = BartTokenizer.from_pretrained("Tianlin668/MentalBART")
    bart_model = BartForConditionalGeneration.from_pretrained("Tianlin668/MentalBART")
    os.makedirs(mental_bart_dir, exist_ok=True)
    bart_tokenizer.save_pretrained(mental_bart_dir)
    bart_model.save_pretrained(mental_bart_dir)
else:
    bart_tokenizer = BartTokenizer.from_pretrained(mental_bart_dir)
    bart_model = BartForConditionalGeneration.from_pretrained(mental_bart_dir)

bart_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model = bart_model.to(bart_device)

def generate_mentalbart_response(input_texts):
    combined_text = " ".join(input_texts)
    inputs = bart_tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {key: value.to(bart_device) for key, value in inputs.items()}
    outputs = bart_model.generate(**inputs, max_length=1000, num_beams=5, early_stopping=True)
    return bart_tokenizer.decode(outputs[0], skip_special_tokens=True)

mentalbart_response = generate_mentalbart_response(texts)

response_split = mentalbart_response.split("Reasoning:")
if len(response_split) > 1:
    response = response_split[0].strip()
    reasoning = "Reasoning:" + response_split[1].strip()
else:
    response = mentalbart_response.strip()
    reasoning = ""

# Response Part
story.append(Paragraph(response, styles['NormalStyle']))

# Add Spacer
story.append(Spacer(1, 20))

# Reasoning Part
if reasoning:
    story.append(Paragraph(reasoning, styles['NormalStyle']))

# Build PDF
doc.build(story)
print(f"PDF report generated: {pdf_path}")

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Email configuration
sender_email = "aceswiftsteadyfast@gmail.com"
sender_password = "yhry hpqh udmx drls"
receiver_email = "garvkumar68@gmail.com"
subject = "Mental Health Report"
body = "The PDF of the mental health report is as follows:"

# Email message setup
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = subject

# Attach email body
msg.attach(MIMEText(body, 'plain'))

# Attach PDF
pdf_path = "Sentiment_Analysis_Report.pdf"
with open(pdf_path, "rb") as attachment:
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename={os.path.basename(pdf_path)}",
    )
    msg.attach(part)

# Send email
try:
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()  # Start TLS encryption
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"Email sent successfully to {receiver_email}")
except Exception as e:
    print(f"Error occurred while sending email: {e}")
