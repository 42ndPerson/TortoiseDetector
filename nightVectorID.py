import torch
from PIL import Image
import open_clip
import numpy as np
import cv2
import os

plant_only_cam_vid_paths = {
    'cam1': {
        'day': '/Volumes/Tortise/cam1/cam1-capture-00000000.mp4',
        'night': '/Volumes/Tortise/cam1/cam1-capture-00000004.mp4'
    },
    'cam2': {
        'day': '/Volumes/Tortise/cam2/cam2-capture-00000001.mp4',
        'night': '/Volumes/Tortise/cam2/cam2-capture-00000002.mp4'
    },
    'cam3': {
        'day': '/Volumes/Tortise/cam3/20250625-075235/cam3-capture-00000000.mp4',
        'night': '/Volumes/Tortise/cam3/20250625-075235/cam3-capture-00000026.mp4'
    },
    'cam4': {
        'day': '/Volumes/Tortise/cam4/cam4-capture-00000000.mp4',
        'night': '/Volumes/Tortise/cam4/cam4-capture-00000017.mp4'
    },
    'cam6': {
        'day': '/Volumes/Tortise/cam6/cam6-capture-00000000.mp4',
        'night': '/Volumes/Tortise/cam6/cam6-capture-00000017.mp4'
    },
}

# Model Setup
model_name = 'PE-Core-bigG-14-448'#'ViT-g-14'#'ViT-B-32'
checkpoint_name = 'meta'#'laion2b_s34b_b88k'#'laion2b_s34b_b79k'
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=checkpoint_name)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active

def get_embd_vec_for_file(file):
    # Video Loading and Frame Extraction
    vid = cv2.VideoCapture(file)

    if not vid.isOpened(): # Check if the video file was opened successfully
        print("Error: Could not open video file.")
    else:
        vid.set(cv2.CAP_PROP_POS_FRAMES, 10)
    
        # Read a frame from the video
        ret, raw_frame = vid.read()

        # If ret is False, it means there are no more frames to read (end of video)
        if not ret:
            print("End of video or error reading frame.")
            sys.exit()
            
        # Crop Image
        crop_frame = raw_frame[75:, :, :]
            
        # Resize, reorder channels, unsqueeze, and torch
        rgb_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)
        frame = preprocess(pil_frame).unsqueeze(0)
        
        # Release the VideoCapture object and destroy all OpenCV windows
        vid.release()
        cv2.destroyAllWindows()
        
        # Classify
        with torch.no_grad(), torch.autocast(device_type="mps"):
            image_features = model.encode_image(frame)
#            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            return image_features

# Calc night vecs
night_vecs = {}
for cam in plant_only_cam_vid_paths:
    day = get_embd_vec_for_file(plant_only_cam_vid_paths[cam]['day'])
    night = get_embd_vec_for_file(plant_only_cam_vid_paths[cam]['night'])
    
    night_vecs[cam] = night - day

# Validate Info
print(torch.nn.functional.cosine_similarity(night_vecs['cam1'], night_vecs['cam6'], dim=-1))
for cam in night_vecs:
    print(torch.linalg.vector_norm(night_vecs[cam]))

# Save average night vec
avg_night = torch.zeros(night_vecs['cam1'].shape)
for cam in night_vecs:
    avg_night += night_vecs[cam]
avg_night /= 5
print("Avg Norm: ", torch.linalg.vector_norm(avg_night))
print("Avg Similarity: ", torch.nn.functional.cosine_similarity(avg_night, night_vecs['cam4'], dim=-1))

torch.save(avg_night, f'Average Night Vector ({model_name}).pt')
