import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from typing import Union, Optional
from transformers import AutoProcessor, AutoModel


# Base class for all embedders
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
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_pretrained(model_name_or_path)
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


# TULIP
class TULIPEmbedder(BaseImageEmbedder):
    def __init__(
        self,
        model_name_or_path: str = "TULIP-so400m-14-384",
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(model_name_or_path, device)
        try:
            import open_clip
        except ImportError:
            raise ImportError("Please install open_clip package to use TULIP models.")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name_or_path, pretrained=True
        )
        self.model.to(self.device).eval()

    def embed(self, image: str | Image.Image) -> np.ndarray:
        image = self.preprocess_image(image)
        if image is None:
            raise ValueError("Invalid image input for embedding.")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = F.normalize(image_features, p=2, dim=1)
        return image_features.cpu().numpy().flatten()


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
