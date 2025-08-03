import torch
from PIL import Image
import open_clip

# Prep Torch MPS
torch_acc_code = 'mps'
device = torch.device(torch_acc_code)

# Model Summary
# PE-Core-bigG-14-448 (Checkpoint: meta)
#   Highly accurate and does not need night correction but is very slow (runs at higher resolution of 448x448)
#   Catches even obscure tortises and synthetic night tortise
#   Test with synthetic data suggests that night correction would remove its ability to detect tortises at night
# ViT-g-14 (Checkpoint: laion2b_s34b_b88k)
#   Intermediate model that somehow performs worse than the light model
#   Needs night correction to not false positive on all night samples, but likely looses ability to detect night tortises
#   Cannot detect synthetic night tortise with night correction and sometimes fails at night/day differentiation
# ViT-B-32 (Checkpoint: laion2b_s34b_b79k)
#   Light model that doesn't seem to produce many false negatives, but produces many false positives
#   Needs night correction to not false positive on all night samples, but likely looses ability to detect night tortises
# Note: Forcing a larger image size for smaller models doesn't seem to lead to a improvement at first blush

# Model Setup
model_name = 'ViT-B-32'#'PE-Core-bigG-14-448'#'ViT-g-14'#'ViT-B-32'
checkpoint_name = 'laion2b_s34b_b79k'#'meta'#'laion2b_s34b_b88k'#'laion2b_s34b_b79k'
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=checkpoint_name, device=device)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer(model_name)

# Data
images = [preprocess(Image.open("Desert.png")).unsqueeze(0), preprocess(Image.open("NightDesert.png")).unsqueeze(0), preprocess(Image.open("Tortise.png")).unsqueeze(0), preprocess(Image.open("ObscureTortise.png")).unsqueeze(0), preprocess(Image.open("ObscureTortise2.png")).unsqueeze(0), preprocess(Image.open("GeminiSyntheticNightTortise.png")).unsqueeze(0)]
target_text = tokenizer(["empty desert (plants)", "tortise"]).to(device)
day_night_text = tokenizer(["day", "night vision"]).to(device)

# Night Correction
enable_night_correction = True
night_vec = torch.load(f'Average Night Vector ({model_name}).pt').to(device)

with torch.no_grad(), torch.autocast(device_type="mps"):
    for image in images:
        # Encoding
        image_features = model.encode_image(image.to(device))
        day_night_text_features = model.encode_text(day_night_text)
        target_text_features = model.encode_text(target_text)
        
        # ID Day and Night
        raw_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        day_night_text_features /= day_night_text_features.norm(dim=-1, keepdim=True)
        day_night_text_probs = (100.0 * raw_image_features @ day_night_text_features.T).softmax(dim=-1)
        
        # Correct Night Photos
        # Model seems to be thrown off by the infrared night vision feed, subtracting average night vector to compensate
        day_or_night = 'D'
        if day_night_text_probs[0][0] < day_night_text_probs[0][1]:
            day_or_night = 'N'
            if enable_night_correction:
                image_features -= 1.5*night_vec # 1.5 is a guess, but some up scaling is apprpriate as night_vec is an average of individual night_vecs, giving it a lower magnitude than the average of the individual magnitudes
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        target_text_features /= target_text_features.norm(dim=-1, keepdim=True)

        target_text_probs = (100.0 * image_features @ target_text_features.T).softmax(dim=-1)
        print("Label probs:", target_text_probs, day_or_night)
