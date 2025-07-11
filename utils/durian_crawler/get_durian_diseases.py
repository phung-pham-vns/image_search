import os
import json
import time
import openai
import logging
from pathlib import Path
from metadata.durian_diseases import durian_diseases
from dotenv import load_dotenv


load_dotenv("/Users/mac/Documents/PHUNGPX/MLOps_practice/durian_crawler/.env")


# ✅ Set your OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or hardcode like: "sk-..."

# 📁 Output folder
save_dir = Path("outputs/durian_diseases")
save_dir.mkdir(parents=True, exist_ok=True)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)


# 🧠 Prompt builder
def build_durian_diseases_prompt(disease_name):
    return f"""
You are a subject-matter expert in Thai agricultural products. Create a comprehensive, accurate, and well-researched profile of the Thai durian disease named "{disease_name}". Format your response strictly as a JSON object following the schema provided below.

Ensure each field contains detailed and up-to-date information based on reputable sources, such as Thai government reports, academic research, agricultural extension publications, or industry data. Cite all facts, figures, and claims using credible references in the `references` array.

JSON Schema:
{{
  "disease": "Official Thai name of the disease (include Thai script and Romanized transliteration)",
  "english_translation": "Commonly used English name of the disease",
  "scientific_name": "Scientific name of the disease-causing organism or condition",
  "description": "Concise overview of the disease and its significance",
  "symptoms": "Visible signs or symptoms observed in durian trees or fruits",
  "effects": "Impact of the disease on durian health, quality, or yield",
  "causes": "Known causes or contributing environmental/agricultural factors",
  "treatment": "Effective treatment methods or management practices",
  "prevention": "Preventative measures or best practices to reduce disease risk",
  "pathogens": "Specific bacteria, fungi, viruses, or pests responsible (if applicable)",
  "locations": "Geographic areas or environmental conditions where the disease is prevalent, with brief justification",
  "durians": "Durian varieties most commonly affected, ranked by susceptibility",
  "references": ["List of credible sources (URLs, publications, or research papers) supporting the information provided"]
}}
"""


# 🚀 Start generating
for i, durian_disease in enumerate(durian_diseases, 1):
    print(f"[{i}/{len(durian_diseases)}] Processing: {durian_disease}")

    try:
        prompt = build_durian_diseases_prompt(durian_disease)
        start_time = time.time()

        # 🔗 OpenAI API call
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        elapsed = time.time() - start_time
        print(f"\t⏱️ Completed in {elapsed:.2f} seconds")

        # 🧹 Clean and validate JSON
        content = response["choices"][0]["message"]["content"].strip().strip("`json \n")
        try:
            data = json.loads(content)
            print("\t✅ JSON is valid")

            # 📁 Save JSON to file
            save_name = (
                durian_disease.replace("(", "")
                .replace(")", "")
                .replace(" ", "_")
                .replace("/", "_")
            )
            file_path = save_dir / f"{i}_{save_name}.json"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"\t📦 Saved to {file_path}")

        except json.JSONDecodeError as e:
            print("\t❌ Invalid JSON response")
            print(f"\t🔎 Error: {e}")

    except Exception as e:
        print(f"❌ API Error for '{durian_disease}': {e}")
