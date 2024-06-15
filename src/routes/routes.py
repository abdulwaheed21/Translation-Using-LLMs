from langdetect import detect
from flask import Flask, render_template, request, jsonify
from src import app
from src.controllers.controller import preprocess_text, translate_text, en_tokenizer, en_model, ur_model, ur_tokenizer
@app.route('/')
def index():
    return render_template('main.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        text = data['text']
        
        # Detect the language of the input text
        lang = detect(text)
        
        # Preprocess text
        text = preprocess_text(text)
        
        # Translate text based on detected language
        if lang == 'en':
            translation = translate_text(text, en_tokenizer, en_model)
        elif lang == 'ur':
            translation = translate_text(text, ur_tokenizer, ur_model)
        else:
            return jsonify({'error': 'Unsupported language'})

        return jsonify({'translation': translation})

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route("/testimonials")
def service():
    return render_template("testimonials.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/team")
def team_members():
    return render_template("team.html")

@app.route("/model")
def model():
    return render_template("index.html")