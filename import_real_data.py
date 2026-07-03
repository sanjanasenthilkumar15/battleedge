import os
import glob
import shutil
import random
from data.generate_synthetic import _spectrogram_png

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
IMG_DIR = os.path.join(DATA_DIR, "images")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

# 1. Clear existing generated images and audio
for f in glob.glob(os.path.join(IMG_DIR, "*.jpg")):
    os.remove(f)
for f in glob.glob(os.path.join(AUDIO_DIR, "*")):
    os.remove(f)

# 2. Import Audio
normal_src = r"C:\Users\sanga\Downloads\Normal"
abnormal_src = r"C:\Users\sanga\Downloads\Abnormal"

normal_files = glob.glob(os.path.join(normal_src, "*.wav"))
abnormal_files = glob.glob(os.path.join(abnormal_src, "*.wav"))

print(f"Found {len(normal_files)} normal audio and {len(abnormal_files)} abnormal audio files.")

# Take up to 50 of each
for i, src in enumerate(normal_files[:50]):
    dst_wav = os.path.join(AUDIO_DIR, f"normal_{i:02d}.wav")
    dst_png = os.path.join(AUDIO_DIR, f"normal_{i:02d}.png")
    shutil.copy2(src, dst_wav)
    _spectrogram_png(dst_wav, dst_png)
    
for i, src in enumerate(abnormal_files[:50]):
    dst_wav = os.path.join(AUDIO_DIR, f"anomaly_{i:02d}.wav")
    dst_png = os.path.join(AUDIO_DIR, f"anomaly_{i:02d}.png")
    shutil.copy2(src, dst_wav)
    _spectrogram_png(dst_wav, dst_png)
    
print("Audio import complete.")

# 3. Import Images
yolo_base = r"C:\Users\sanga\Downloads\archive\The Welding Defect Dataset\The Welding Defect Dataset\train"
images_dir = os.path.join(yolo_base, "images")
labels_dir = os.path.join(yolo_base, "labels")

image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
print(f"Found {len(image_files)} images in YOLO dataset.")

defect_types = ["porosity", "burn_through", "contamination", "lack_of_fusion", "spatter", "cold_weld", "misalignment"]
defect_counters = {d: 0 for d in defect_types}
good_weld_count = 0

random.seed(42)
random.shuffle(image_files)

for img_path in image_files:
    basename = os.path.basename(img_path)
    label_path = os.path.join(labels_dir, basename.replace(".jpg", ".txt"))
    
    is_good = False
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            # Check if there is a '1' class (Good Weld)
            if any(line.startswith("1") for line in lines):
                is_good = True
    
    if is_good:
        if good_weld_count < 20: # Keep up to 20 good welds
            dst = os.path.join(IMG_DIR, f"good_weld_{good_weld_count:02d}.jpg")
            shutil.copy2(img_path, dst)
            good_weld_count += 1
    else:
        # Assign to a random defect type to ensure we have samples for all categories
        dtype = random.choice(defect_types)
        if defect_counters[dtype] < 15: # Keep up to 15 of each defect type
            dst = os.path.join(IMG_DIR, f"{dtype}_{defect_counters[dtype]:02d}.jpg")
            shutil.copy2(img_path, dst)
            defect_counters[dtype] += 1

print("Image import complete.")
