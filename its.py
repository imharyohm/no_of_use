from flask import Flask, request, render_template
import os
import base64
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OpenAI API Key - Replace with your key
OPENAI_API_KEY = "sk-UCuBbs9rn2Nx_NQzMgBdqw"

# Initialize Langchain OpenAI client
chat_model = ChatOpenAI(model="azure.gpt-4o-mini", api_key=OPENAI_API_KEY, base_url="https://genai-sharedservice-americas.pwcinternal.com")

# Supported categories
VALID_CATEGORIES = ["abusive", "spam", "hate speech", "harmless"]

# Convert image to base64 for OpenAI
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Analyze image using GPT-4 Turbo Vision
def analyze_image_with_openai(image_path):
    image_base64 = image_to_base64(image_path)

    prompt = """
    You are a content moderation assistant. 
    Analyze the provided image and classify the content into one of these categories: 
    ["abusive", "spam", "hate speech", "harmless"]. 
    Only return the category, nothing else.
    """

    message = [
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            ]
        )
    ]

    response = chat_model.invoke(message)
    category = response.content.strip().lower()

    if category not in VALID_CATEGORIES:
        category = "unknown"

    return category

# Analyze text using GPT-4 Turbo Text
def analyze_text_with_openai(text):
    prompt = f"""
    You are a content moderation assistant. 
    Analyze the following text and classify it into one of these categories: 
    ["abusive", "spam", "hate speech", "harmless"]. 
    Only return the category, nothing else.

    Text: {text}
    """

    message = [HumanMessage(content=prompt)]
    response = chat_model.invoke(message)
    category = response.content.strip().lower()

    if category not in VALID_CATEGORIES:
        category = "unknown"

    return category

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        analysis_type = request.form.get('analysis_type')

        if analysis_type == 'text':
            input_text = request.form.get('text')
            if input_text:
                category = analyze_text_with_openai(input_text)
                result = {"input_type": "text", "content": input_text, "category": category}

        elif analysis_type == 'image':
            if 'file' not in request.files:
                return "No file part"

            file = request.files['file']
            if file.filename == '':
                return "No file selected"

            if file:
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)

                category = analyze_image_with_openai(file_path)
                result = {"input_type": "image", "filename": file.filename, "category": category}

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
