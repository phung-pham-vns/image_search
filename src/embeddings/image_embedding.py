import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from typing import Union
from transformers import AutoProcessor, AutoModel


class ImageEmbedding:
    def __init__(
        self,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        device: Union[str, torch.device] = "cpu",
    ):
        try:
            self.device = torch.device(device)
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_pretrained(model_name_or_path)
            self.model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model or processor: {e}")

    def preprocess_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB format if necessary."""
        try:
            image = Image.open(image_path)
            return image.convert("RGB") if image.mode != "RGB" else image
        except Exception as e:
            print(f"Error processing image at '{image_path}': {e}")
            return None

    def embed(self, image: str | Image.Image) -> np.ndarray:
        """Embed an image and return a normalized feature vector."""
        try:
            if isinstance(image, str):
                image = self.preprocess_image(image)
                if not image:
                    return None

            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = F.normalize(image_features, p=2, dim=1)

            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Failed to embed image: {e}")
            return None
