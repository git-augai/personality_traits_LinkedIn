import streamlit as st
import json
import torch
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings
import os
import toml
from groq import Groq

# Suppress warnings
warnings.filterwarnings("ignore")


# Load secrets from a default or user-specified path
def load_secrets(default_path=None):
    """
    Load secrets from a TOML file located at `default_path` or a user-specified path.
    """
    path = default_path or r"./secrets.toml"  # Default to the current working directory
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Secrets file not found at {path}")
        with open(path, "r") as file:
            return toml.load(file)
    except Exception as e:
        st.error(f"Failed to load secrets: {e}")
        return None


# Function to extract LinkedIn profile data
def extract_linkedin_profile_data(url, api_key):
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(api_endpoint, headers=headers, params={"url": url})

        if response.status_code == 200:
            profile_data = response.json()
            return profile_data
        else:
            st.error(f"Error: Unable to fetch data. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# Function to Transform Strategies to Single String
def transform_strategies_to_single_string(strategies):
    transformed_strategies = {}
    for key, value in strategies.items():
        if isinstance(value, list):
            transformed_strategies[key] = " ".join(value)
        else:
            transformed_strategies[key] = value
    return transformed_strategies


# Personality Trait Analysis Function
def analyze_personality_traits(text):
    model = AutoModelForSequenceClassification.from_pretrained(
        "KevSun/Personality_LM", ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained("KevSun/Personality_LM")

    encoded_input = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=64
    )

    model.eval()

    with torch.no_grad():
        outputs = model(**encoded_input)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_scores = predictions[0].tolist()

    trait_names = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]

    trait_scores = {trait: score for trait, score in zip(trait_names, predicted_scores)}
    return trait_scores


# Function to Generate Cold Calling Report using Groq
def generate_cold_calling_report_groq(personality_traits, groq_api_key):
    # Initialize Groq client
    client = Groq(api_key=groq_api_key)

    # Create prompt for cold calling strategy
    prompt = f"""
    Based on the following Big Five personality traits:
    - Agreeableness: {personality_traits['agreeableness']:.2f}
    - Openness: {personality_traits['openness']:.2f}
    - Conscientiousness: {personality_traits['conscientiousness']:.2f}
    - Extraversion: {personality_traits['extraversion']:.2f}
    - Neuroticism: {personality_traits['neuroticism']:.2f}

    Provide a detailed cold-calling strategy report in JSON format with these keys:
    - Pattern Interrupt: [suggestions based on personality traits]
    - Pace: [suggestions based on personality traits]
    - Tone: [suggestions based on personality traits]
    - Tactics to Win: [suggestions based on personality traits]
    - Mistakes to Avoid: [suggestions based on personality traits]
    - Making the Ask: [suggestions based on personality traits]
    - Subconscious Driver: [suggestions based on personality traits]

    Each key should have a strategy tailored to these personality traits. 
    Return ONLY the valid JSON response.
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gemma2-9b-it",
        response_format={"type": "json_object"},
    )

    return json.loads(chat_completion.choices[0].message.content)


# Main Streamlit application
def main():
    st.title("Cold Calling Strategy Generator")
    st.write("Analyze LinkedIn profiles and generate cold calling strategies.")

    # Specify the default path to the secrets file
    default_secrets_path = r"./secrets.toml"  # Or any other standard location
    secrets = load_secrets(default_secrets_path)
    if not secrets:
        return

    PROXYCURL_API_KEY = secrets["api_keys"]["proxycurl"]
    GROQ_API_KEY = secrets["api_keys"]["groq"]

    linkedin_url = st.text_input("Enter LinkedIn Profile URL")

    if st.button("Fetch LinkedIn Profile"):
        if linkedin_url:
            profile_data = extract_linkedin_profile_data(linkedin_url, PROXYCURL_API_KEY)
            if profile_data:
                st.success("LinkedIn profile data fetched successfully!")
                st.json(profile_data)
            else:
                st.error("Failed to fetch LinkedIn profile data. Check the URL.")
        else:
            st.warning("Please provide the LinkedIn URL.")

    if st.button("Generate Cold Calling Strategy"):
        profile_data = extract_linkedin_profile_data(linkedin_url, PROXYCURL_API_KEY)
        if profile_data:
            analysis_text = f"""
            Occupation: {profile_data.get('occupation', '')}
            Headline: {profile_data.get('headline', '')}
            Summary: {profile_data.get('summary', '')}

            Experiences:
            """
            for exp in profile_data.get("experiences", []):
                analysis_text += f"Title: {exp.get('title', 'N/A')}\n"
                analysis_text += f"Description: {exp.get('description', 'N/A')}\n\n"

            st.write("Analyzing personality traits...")
            personality_traits = analyze_personality_traits(analysis_text)

            st.write("Generating cold calling strategy...")
            cold_calling_strategies = generate_cold_calling_report_groq(personality_traits, GROQ_API_KEY)

            cold_calling_strategies = transform_strategies_to_single_string(cold_calling_strategies)

            st.json(cold_calling_strategies)
        else:
            st.error("No profile data found. Fetch LinkedIn profile first.")


if __name__ == "__main__":
    main()