import os
import torch
import httpx
import numpy as np
import torch.nn.functional as F

from PIL import Image
from typing import Union, Optional
from transformers import AutoProcessor, AutoModel
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class BaseImageEmbedder:
    def __init__(
        self,
        model_name_or_path: str,
        device: Union[str, torch.device] = "cpu",
    ):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)

    def preprocess_image(self, image: str | Image.Image) -> Optional[Image.Image]:
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except Exception as e:
                print(f"Error processing image at '{image}': {e}")
                return None
        # Handle UploadedFile objects from Streamlit
        elif hasattr(image, "read"):  # UploadedFile has a read method
            try:
                image = Image.open(image)
            except Exception as e:
                print(f"Error processing uploaded file: {e}")
                return None
        return image.convert("RGB") if image.mode != "RGB" else image

    def embed(self, image: str | Image.Image) -> np.ndarray:
        raise NotImplementedError


# CLIP implementation
class CLIPEmbedder(BaseImageEmbedder):
    def __init__(
        self,
        model_name_or_path: str,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(model_name_or_path, device)
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path, token=HF_TOKEN
            )
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                token=HF_TOKEN,
            )
            self.model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CLIP model: {e}")

    def embed(self, image: str | Image.Image) -> np.ndarray:
        image = self.preprocess_image(image)
        if image is None:
            raise ValueError("Invalid image input for embedding.")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=1)
        return image_features.cpu().numpy().flatten()


# SigLIP implementation (uses same interface as CLIP)
class SigLIPEmbedder(CLIPEmbedder):
    pass  # For now, SigLIP models are loaded the same way as CLIP


# DINOv2 implementation
class DINOv2Embedder(BaseImageEmbedder):
    def __init__(
        self, model_name_or_path: str, device: Union[str, torch.device] = "cpu"
    ):
        super().__init__(model_name_or_path, device)
        try:
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_pretrained(model_name_or_path)
            self.model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DINOv2 model: {e}")

    def embed(self, image: str | Image.Image) -> np.ndarray:
        image = self.preprocess_image(image)
        if image is None:
            raise ValueError("Invalid image input for embedding.")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model(**inputs).last_hidden_state[:, 0]
            image_features = F.normalize(image_features, p=2, dim=1)
        return image_features.cpu().numpy().flatten()


class TULIPEmbedder:
    """
    Central registry of embedding-capable models.
    """

    def __init__(self, image_embedder_url, image_embedder_token):
        self.base_url = image_embedder_url
        self.api_key = image_embedder_token
        self.endpoint = "image_embeddings"

    async def embedding(self, image: str):
        """Embed a list of texts using a specified model."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": "string", "input": [image]}

        response = await httpx.AsyncClient().post(
            f"{self.base_url}{self.endpoint}", headers=headers, json=payload, timeout=30
        )

        response_json = response.json()

        embedding = response_json["data"][0]["embedding"]

        return embedding


# Main selector class
class ImageEmbedding:
    def __init__(
        self,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        device: Union[str, torch.device] = "cpu",
    ):
        # Model selection logic
        model_name = model_name_or_path.lower()
        if any(x in model_name for x in ["clip", "openai/clip"]):
            self.backend = CLIPEmbedder(model_name_or_path, device)
        elif "siglip" in model_name:
            self.backend = SigLIPEmbedder(model_name_or_path, device)
        elif "dino" in model_name:
            self.backend = DINOv2Embedder(model_name_or_path, device)
        elif "tulip" in model_name:
            self.backend = TULIPEmbedder(model_name_or_path, device)
        else:
            # Default to CLIP-like interface for unknown models
            self.backend = CLIPEmbedder(model_name_or_path, device)

    def preprocess_image(self, image: str | Image.Image) -> Optional[Image.Image]:
        return self.backend.preprocess_image(image)

    def embed(self, image: str | Image.Image) -> np.ndarray:
        return self.backend.embed(image)


if __name__ == "__main__":
    embedder = ImageEmbedding(model_name_or_path="google/siglip2-base-patch16-224")
    img = "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/processed_dataset/disease/images/0af2a5be-1ef6-420c-b86a-4f46c19b244a.jpeg"  # Replace with your image path
    embedding = embedder.embed(img)
    print("Image embedding shape:", embedding.shape)
    print("Embedding:", embedding)
