import google.generativeai as genai
import os
# Configure your API key
genai.configure(api_key=os.environ["GEMINI_API_TOKEN"])  # or use environment variable

# Function to generate a response
def run(prompt):
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "stop_sequences": ["***"],
            "max_output_tokens": 100,
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 1,
        }
    )

    response = model.generate_content(prompt)
    return response.text

# Example usage

