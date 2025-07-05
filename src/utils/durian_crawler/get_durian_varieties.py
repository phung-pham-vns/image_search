import os
import json
import time
import openai
import logging
from pathlib import Path
from metadata.durian_varieties import durian_varieties
from dotenv import load_dotenv


load_dotenv("/Users/mac/Documents/PHUNGPX/MLOps_practice/durian_crawler/.env")

# ✅ Set your OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or hardcode like: "sk-..."

# 📁 Output folder
save_dir = Path("outputs/durian_varieties")
save_dir.mkdir(parents=True, exist_ok=True)


# 🧠 Prompt builder
def build_durian_varieties_prompt(durian_name):
    return f"""
You are a subject-matter expert in Thai agricultural products. Generate a detailed and well-researched profile of the Thai durian variety "{durian_name}". Format the response as a JSON object that strictly follows the schema below.

Ensure that each field is comprehensive, factually accurate, and reflects the most up-to-date information available. Rely on credible sources such as official Thai government publications, peer-reviewed agricultural studies, export data, and industry reports. Where applicable, provide estimates in both Thai Baht (THB) and U.S. Dollars (USD). Include direct reference links for all statistics, claims, or figures in the `references` array.

JSON Schema:
{{
  "variety_name": "The official Thai name of the durian, including Thai script and Romanized transliteration",
  "english_translation": "The commonly used English name for this durian variety",
  "market_ranking": "This variety’s relative importance and ranking in both the Thai and international durian markets, including an explanation",
  "cultivation_location": "Key provinces or regions in Thailand where this durian is primarily grown",
  "cultivation_percentage": "Estimated percentage share of national durian cultivation for this variety",
  "cultivation_methods": "Typical farming practices and techniques used to cultivate this durian",
  "export_profile": "Main export destinations, export formats (e.g., fresh, frozen), and notable export performance or trends",
  "fruit_specifications": "Comprehensive physical description: average size, weight, color, texture, aroma, flavor profile, flesh-to-seed ratio, shape, and distinctive characteristics",
  "harvest_duration": "Standard growing and harvest timeline by region, noting whether it is early-, mid-, or late-season",
  "historical_pricing": "Estimated price ranges (in THB and USD) over the past three years, including variations by grade or season if available",
  "references": ["List of source URLs or full citations used to support the data provided"]
}}
"""


# 🚀 Start generating
for i, durian in enumerate(durian_varieties, 1):
    print(f"[{i}/{len(durian_varieties)}] Processing: {durian}")

    try:
        prompt = build_durian_varieties_prompt(durian)
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
            save_name = durian.replace("(", "").replace(")", "").replace(" ", "_")
            file_path = save_dir / f"{i}_{save_name}.json"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"\t📦 Saved to {file_path}")

        except json.JSONDecodeError as e:
            print("\t❌ Invalid JSON response")
            print(f"\t🔎 Error: {e}")

    except Exception as e:
        print(f"❌ API Error for '{durian}': {e}")
