import base64
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    # Get prompt from request
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "Prompt parameter is required"}), 400
    
    try:
        # Initialize Gemini client
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        
        model = "gemini-2.0-pro-exp-02-05"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=64,
            max_output_tokens=8192,
            response_mime_type="text/plain",
        )
        
        # Generate content from Gemini
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        return jsonify({"response": response.text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)