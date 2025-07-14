import os
import torch
import httpx
import open_clip
import numpy as np
import torch.nn.functional as F

from PIL import Image
from typing import Union, Optional
from transformers import AutoProcessor, AutoModel
from dotenv import load_dotenv
from src.constants import EMBEDDING_MODELS

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


ImageType = Union[str, Image.Image]


class BaseImageEmbedder:
    def __init__(
        self, model_name_or_path: str, device: Union[str, torch.device] = "cpu"
    ):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)

    def preprocess_image(self, image: ImageType) -> Optional[Image.Image]:
        try:
            if isinstance(image, str):
                image = Image.open(image)
            elif hasattr(image, "read"):
                image = Image.open(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            print(f"[Error] Failed to open image: {e}")
            return None

    def embed(self, image: ImageType) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement the embed method.")


class CLIPEmbedder(BaseImageEmbedder):
    def __init__(
        self, model_name_or_path: str, device: Union[str, torch.device] = "cpu"
    ):
        super().__init__(model_name_or_path, device)
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path, token=HF_TOKEN
            )
            self.model = AutoModel.from_pretrained(model_name_or_path, token=HF_TOKEN)
            self.model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"[CLIP] Failed to initialize model: {e}")

    def embed(self, image: ImageType) -> np.ndarray:
        image = self.preprocess_image(image)
        if image is None:
            raise ValueError("Invalid image input.")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy().flatten()


class SigLIPEmbedder(CLIPEmbedder):
    """Uses the same logic as CLIP."""

    pass


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
            raise RuntimeError(f"[DINOv2] Failed to initialize model: {e}")

    def embed(self, image: ImageType) -> np.ndarray:
        image = self.preprocess_image(image)
        if image is None:
            raise ValueError("Invalid image input.")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model(**inputs).last_hidden_state[:, 0]
            features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy().flatten()


class TULIPEmbedder(BaseImageEmbedder):
    def __init__(
        self, model_name_or_path: str, device: Union[str, torch.device] = "cpu"
    ):
        super().__init__(model_name_or_path, device)
        self.model, _, self.processor = open_clip.create_model_and_transforms(
            model_name_or_path,
            pretrained=EMBEDDING_MODELS[model_name_or_path]["model_path"],
        )
        self.model.to(self.device).eval()

    def preprocess_image(self, image: ImageType) -> Optional[torch.Tensor]:
        image = super().preprocess_image(image)
        if image is None:
            return None
        return self.processor(image).unsqueeze(0).to(self.device)

    def embed(self, image: ImageType) -> np.ndarray:
        image_tensor = self.preprocess_image(image)
        if image_tensor is None:
            raise ValueError("Invalid image input.")
        with torch.no_grad(), torch.autocast("cuda"):
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()


class TULIPAPIEmbedder:
    """
    Calls a remote embedding API to get image embeddings.
    """

    def __init__(self, image_embedder_url: str, image_embedder_token: str):
        self.base_url = image_embedder_url
        self.api_key = image_embedder_token
        self.endpoint = "image_embeddings"

    async def embedding(self, image: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": "string", "input": [image]}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}{self.endpoint}",
                headers=headers,
                json=payload,
                timeout=30,
            )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


class ImageEmbedding:
    def __init__(
        self,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        device: Union[str, torch.device] = "cpu",
    ):
        model_key = model_name_or_path.lower()
        if any(k in model_key for k in ["clip", "openai/clip"]):
            self.backend = CLIPEmbedder(model_name_or_path, device)
        elif "siglip" in model_key:
            self.backend = SigLIPEmbedder(model_name_or_path, device)
        elif "dino" in model_key:
            self.backend = DINOv2Embedder(model_name_or_path, device)
        elif "tulip" in model_key:
            self.backend = TULIPEmbedder(model_name_or_path, device)
        else:
            self.backend = CLIPEmbedder(model_name_or_path, device)

    def preprocess_image(self, image: ImageType) -> Optional[Image.Image]:
        return self.backend.preprocess_image(image)

    def embed(self, image: ImageType) -> np.ndarray:
        return self.backend.embed(image)


if __name__ == "__main__":
    embedder = ImageEmbedding("TULIP-B-16-224")
    img_path = "dataset/images/processed_dataset/disease/images/0af2a5be-1ef6-420c-b86a-4f46c19b244a.jpeg"
    embedding = embedder.embed(img_path)
    print("Image embedding shape:", embedding.shape)
    print("Embedding:", embedding)
