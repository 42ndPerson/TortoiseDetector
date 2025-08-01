import torch
from PIL import Image
import open_clip
import numpy as np
import cv2
import os

# Settings
SAMPLE_INTERVAL = 15 # Seconds

# File Search
def get_all_files_os_walk(root_folder):
    file_names = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            file_names.append(os.path.join(dirpath, filename))
    return file_names
    
all_files = get_all_files_os_walk('/Volumes/Tortise')[:100]
#all_files = list(filter(lambda s: '20250624-204002/cam3-capture-00000012' in s, all_files))#[:20]
print("File Count: " + str(len(all_files)))
# Good Test Files: '20250624-204002/cam3-capture-00000009', '20250624-204002/cam3-capture-00000012'

# Prep Torch MPS
metal_device = torch.device('mps')

# Model Setup
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=metal_device)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')
day_night_text = tokenizer(["day", "night vision"]).to(metal_device)
target_text = tokenizer(["empty", "tortise"]).to(metal_device)

# Retrieve Night Vector
night_vec = torch.load('Average Night Vector.pt').to(metal_device)

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
            vid.set(cv2.CAP_PROP_POS_FRAMES, read_pos-1)
            read_pos += interval
        
            # Read a frame from the video
            ret, raw_frame = vid.read()

            # If ret is False, it means there are no more frames to read (end of video)
            if not ret:
                print("End of video or error reading frame.")
                break
                
            # Crop Image
            crop_frame = raw_frame[75:, :, :]
            
#            cv2.imshow('Video Player', crop_frame) # Display the frame
#            if cv2.waitKey(25) & 0xFF == ord('q'): # Wait for 'q' key to exit
#                break
                
            # Resize, reorder channels, unsqueeze, and torch
            rgb_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            frame = preprocess(pil_frame).unsqueeze(0)
            
            # Send Frame to MPS
            frame = frame.to(metal_device)
            
            # Classify
            with torch.no_grad(), torch.autocast(device_type="mps"):
                # Encoding
                image_features = model.encode_image(frame)
                day_night_text_features = model.encode_text(day_night_text)
                target_text_features = model.encode_text(target_text)
                
                # ID Day and Night
                raw_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                day_night_text_features /= day_night_text_features.norm(dim=-1, keepdim=True)
                day_night_text_probs = (100.0 * raw_image_features @ day_night_text_features.T).softmax(dim=-1)
                
                # Correct Night Photos
                day_or_night = 'D'
                if day_night_text_probs[0][0] < day_night_text_probs[0][1]:
                    day_or_night = 'N'
                    image_features -= 1.5*night_vec # 1.5 is a guess, but some up scaling is apprpriate as night_vec is an average of individual night_vecs, giving it a lower magnitude than the average of the individual magnitudes
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                target_text_features /= target_text_features.norm(dim=-1, keepdim=True)

                target_text_probs = (100.0 * image_features @ target_text_features.T).softmax(dim=-1)
                print("Label probs:", target_text_probs, file, read_pos/fps)  # prints: [[1., 0., 0.]]
                
                if target_text_probs[0][1] > target_text_probs[0][0]:
                    records.append((read_pos, day_or_night))
                
        # Release the VideoCapture object and destroy all OpenCV windows
        vid.release()
#        cv2.destroyAllWindows()
        
        # Save File
        with open('tortise_sightings.txt', 'a') as out_file:
            out_file.write('\n\n' + file + ':')
            for (raw_time, dn) in records:
                time_secs = raw_time // fps
                min = time_secs // 60
                sec = time_secs % 60
                out_file.write(f"\n   {min:02.0f}:{sec:02.0f} ({dn})")
        print("Saved Sightings for " + file)
