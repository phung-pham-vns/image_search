import os
import requests
import json
import random

URL_image_search = "http://localhost:8002/v1/meilisearch/image_upload"


def get_image(image_path):

    with open(image_path, "rb") as f:
        files = {
            "image": (
                f.name,
                f,
                "image/jpeg",
            )  # "image" phải đúng tên tham số server mong đợi
        }
        data = {
            "search_on": "image",
            "top_k": str(10),
            "index": "Disease",
            # "query": ""  # ch/ỗ này lúc nãy bạn để `None` nên không search theo text
        }
        response = requests.post(URL_image_search, params=data, files=files).json()
        print(response)
    return response


# def get_text(query):
#     data = {
#         "search_on": "content",
#         "top_k": 10,
#         "index": "agriculture_v2",
#         "query": str(query),
#         "semantic_ratio": 1.0,
#         "additionalProp1": {},  # chỗ này lúc nãy bạn để `None` nên không search theo text
#     }
#     # It seems you're calling .json() twice here.
#     # requests.post().json() already parses the JSON response.
#     # So, response.json() again would cause an error if response is already a dict.
#     response = requests.post(URL_text_search, json=data)
#     return response.json()  # Corrected to call .json() only once


image_folder_path = "/Users/noaft/Documents/Disease_evalute"

if __name__ == "__main__":
    list_img_path = [
        os.path.join(image_folder_path, image)
        for image in os.listdir(image_folder_path)
    ]
    result = []
    for folder_path in list_img_path:
        if "store" in folder_path.lower():
            continue
        for image_path in os.listdir(folder_path):
            # Corrected splitting for page. Assuming format like "prefix_P123_456.jpeg"
            # The original split("/").split("_") would likely fail as split("/") returns a list of strings.
            # It's better to get the filename first.
            class_name = folder_path.split("/")[-1]
            image_path_ = os.path.join(folder_path, image_path)

            result.append(
                {
                    "image_path": image_path_,
                    "class": class_name,
                    "result": get_image(image_path_),
                }
            )

    output_path = (
        "/Users/noaft/Documents/aisac-rag-fastapi-data/test/search_disease_results.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
#
