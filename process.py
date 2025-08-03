import torch
from PIL import Image
import open_clip
import numpy as np
import cv2
import os
import random

# Settings
SRC_DIR = '/Volumes/Tortoise'
SAMPLE_INTERVAL = 15 # Seconds
SAVE_TRAINING_DATA = False
PREVIEW_FRAMES = False

# File Search
def get_all_files_os_walk(root_folder):
    file_names = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            file_names.append(os.path.join(dirpath, filename))
    return file_names
    
all_files = get_all_files_os_walk(SRC_DIR)
#all_files = list(filter(lambda s: 'cam2/cam2-capture-00000001' in s, all_files))#[:20]
print("File Count: " + str(len(all_files)))
# Good Test Files: '20250624-204002/cam3-capture-00000009', '20250624-204002/cam3-capture-00000012', 'cam2/cam2-capture-00000001'

# Prep Torch MPS
torch_acc_code = None
if torch.cuda.is_available():
    torch_acc_code = 'cuda'
elif torch.backends.mps.is_available():
    torch_acc_code = 'mps'
else:
    torch_acc_code = 'cpu'
device = torch.device(torch_acc_code)

# Model Setup
model_name = 'PE-Core-bigG-14-448'
checkpoint_name = 'meta'
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=checkpoint_name, device=device)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer(model_name)
day_night_text = tokenizer(["day", "night vision"]).to(device)
target_text = tokenizer(["empty desert (plants)", "tortoise"]).to(device)

# Retrieve Night Vector
enable_night_correction = False
night_vec = torch.load(f'Average Night Vector ({model_name}).pt').to(device)

# Process Videos
for file in all_files:
    # Video Loading and Frame Extraction
    vid = cv2.VideoCapture(file)
    
    # Sighting record
    records = []

    if not vid.isOpened(): # Check if the video file was opened successfully
        print("Error: Could not open video file.")
    else:
        frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vid.get(cv2.CAP_PROP_FPS)
        interval = fps*SAMPLE_INTERVAL
        
        read_pos = 0

        while read_pos < frame_count:
            vid.set(cv2.CAP_PROP_POS_FRAMES, read_pos-1) # Set cv2 for frame read
            time_secs = read_pos // fps # Extract real time units
            min = time_secs // 60
            sec = time_secs % 60
            read_pos += interval # Advance for next frame
        
            # Read a frame from the video
            ret, raw_frame = vid.read()

            # If ret is False, it means there are no more frames to read (end of video)
            if not ret:
                print("End of video or error reading frame.")
                break
                
            # Crop Image
            crop_frame = raw_frame[75:, :, :]
            
            if PREVIEW_FRAMES:
                cv2.imshow('Video Player', crop_frame) # Display the frame
                if cv2.waitKey(25) & 0xFF == ord('q'): # Wait for 'q' key to exit
                    break
                
            # Resize, reorder channels, unsqueeze, and torch
            rgb_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            frame = preprocess(pil_frame).unsqueeze(0)
            
            # Send Frame to MPS
            frame = frame.to(device)
            
            # Classify
            with torch.no_grad(), torch.autocast(device_type=torch_acc_code):
                # Encoding
                image_features = model.encode_image(frame)
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
                print("Label probs:", target_text_probs, file, read_pos/fps)
                
                # Record results and save training data
                if target_text_probs[0][1] > target_text_probs[0][0]:
                    records.append((time_secs, day_or_night))
                    if SAVE_TRAINING_DATA:
                        save_dir = os.path.join('TortoiseImages', day_or_night)
                        os.makedirs(save_dir, exist_ok=True)
                        path = os.path.join(save_dir, f'{os.path.basename(file)}_{min:02.0f}-{sec:02.0f}.png')
                        cv2.imwrite(path, crop_frame)
                else:
                    if SAVE_TRAINING_DATA and random.randint(1, 20) == 10: # Save 1 in 25 desert images
                        save_dir = os.path.join('DesertImage', day_or_night)
                        os.makedirs(save_dir, exist_ok=True)
                        path = os.path.join(save_dir, f'{os.path.basename(file)}_{min:02.0f}-{sec:02.0f}.png')
                        cv2.imwrite(path, crop_frame)
                
        # Release the VideoCapture object and destroy all OpenCV windows
        vid.release()
        if PREVIEW_FRAMES:
            cv2.destroyAllWindows()
        
        # Save File
        with open('tortoise_sightings.txt', 'a') as out_file:
            out_file.write('\n\n' + file + ':')
            for (time_secs, dn) in records:
                min = time_secs // 60
                sec = time_secs % 60
                out_file.write(f"\n   {min:02.0f}:{sec:02.0f} ({dn})")
        print("Saved Sightings for " + file)
