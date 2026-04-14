import cv2
import torch
import clip
from PIL import Image

# -------------------------------
# Load CLIP model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 👉 Keep video in same folder
video_path = "input.mp4"

# -------------------------------
# Extract frames with timestamps
# -------------------------------
def extract_frames(video):
    cap = cv2.VideoCapture(video)

    frames = []
    times = []

    if not cap.isOpened():
        print("❌ Video not found or cannot open!")
        return frames, times

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # fallback

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # take 1 frame per second
        if frame_id % int(fps) == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            times.append(frame_id / fps)

        frame_id += 1

    cap.release()
    return frames, times


# -------------------------------
# Get frame embeddings
# -------------------------------
def get_frame_embeddings(frames):
    embs = []
    for f in frames:
        img = preprocess(f).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img)
        embs.append(emb)

    return torch.cat(embs)


# -------------------------------
# MAIN
# -------------------------------

# Step 1: Extract frames
frames, times = extract_frames(video_path)

print("Total frames extracted:", len(frames))

# Safety check
if len(frames) == 0:
    print("❌ No frames extracted. Check video path or format.")
    exit()

# Step 2: Frame embeddings
frame_embs = get_frame_embeddings(frames)

# Step 3: Text query
query = input("Enter query (e.g. 'car', 'horse'): ")

# Step 4: Text embedding
text = clip.tokenize([query]).to(device)
with torch.no_grad():
    text_emb = model.encode_text(text)

# -------------------------------
# Normalize embeddings (IMPORTANT)
# -------------------------------
frame_embs = frame_embs / frame_embs.norm(dim=1, keepdim=True)
text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)

# Step 5: Similarity
similarity = (frame_embs @ text_emb.T).squeeze().cpu().numpy()

# Step 6: Threshold
threshold = 0.2

print("\nMatching timestamps:\n")

found = False
for i, score in enumerate(similarity):
    if score > threshold:
        print(f"Time: {times[i]:.2f} sec | Score: {score*100:.2f}%")
        found = True

# Always show best match
best_idx = similarity.argmax()
print(f"\nBest match at {times[best_idx]:.2f} sec | Score: {similarity[best_idx]*100:.2f}%")

if not found:
    print("\n(No strong match above threshold, showing best possible match)")