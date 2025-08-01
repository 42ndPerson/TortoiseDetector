import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

images = [preprocess(Image.open("Desert.png")).unsqueeze(0), preprocess(Image.open("NightDesert.png")).unsqueeze(0), preprocess(Image.open("Tortise.png")).unsqueeze(0), preprocess(Image.open("ObscureTortise.png")).unsqueeze(0), preprocess(Image.open("ObscureTortise2.png")).unsqueeze(0)]
text = tokenizer(["desert (plants)", "tortise"])

with torch.no_grad(), torch.autocast(device_type="mps"):
    for image in images:
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
