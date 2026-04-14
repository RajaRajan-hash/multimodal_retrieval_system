import os, torch, clip, cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

video_folder = "database_video"
query_folder = "query"

# -------------------------------
# Extract frames from video
# -------------------------------
def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // num_frames, 1)

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames


# -------------------------------
# Image embedding
# -------------------------------
def get_image_emb(img):
    img = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
    return emb


# -------------------------------
# Video embedding (average frames)
# -------------------------------
def get_video_emb(video_path):
    frames = extract_frames(video_path)
    embs = [get_image_emb(f) for f in frames]
    return torch.mean(torch.cat(embs), dim=0, keepdim=True)


# -------------------------------
# Build video database
# -------------------------------
db_emb, names = [], []

for f in os.listdir(video_folder):
    if f.lower().endswith((".mp4", ".avi", ".mov")):
        path = os.path.join(video_folder, f)
        emb = get_video_emb(path)
        db_emb.append(emb)
        names.append(f)

db_emb = torch.cat(db_emb)


# -------------------------------
# Get query (image OR video)
# -------------------------------
query_path = None
for f in os.listdir(query_folder):
    if f.lower().endswith((".jpg", ".png", ".jpeg", ".mp4", ".avi")):
        query_path = os.path.join(query_folder, f)
        break


# -------------------------------
# Query embedding
# -------------------------------
if query_path.endswith((".jpg", ".png", ".jpeg")):
    img = Image.open(query_path).convert("RGB")
    q_emb = get_image_emb(img)
else:
    q_emb = get_video_emb(query_path)


# -------------------------------
# Similarity
# -------------------------------
scores = cosine_similarity(q_emb.cpu(), db_emb.cpu())[0]
i = scores.argmax()

print("Query:", os.path.basename(query_path))
print("Best matched video:", names[i])
print("Similarity:", scores[i] * 100)   