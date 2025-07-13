import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    "TULIP-B-16-224", pretrained="models/open_clip/tulip-B-16-224.ckpt"
)
model.eval()

image = preprocess(
    Image.open("models/open_clip/images/iStock-1052880600-1024x683.jpg")
).unsqueeze(0)
tokenizer = open_clip.get_tokenizer("TULIP-so400m-14-384")
text = tokenizer(["a cat", "a dog", "a bird"])

with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    print(image_features.shape)
    text_features = model.encode_text(text)
    print(text_features.shape)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probabilities:", similarities)
