import os, torch, clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

db_folder = "database_images"
query_folder = "query"

db_emb, names = [], []

for f in os.listdir(db_folder):
    if f.lower().endswith((".jpg", ".png", ".jpeg")):
        img = preprocess(
            Image.open(os.path.join(db_folder, f)).convert("RGB")
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            db_emb.append(model.encode_image(img))
        names.append(f)

query_image = None
for f in os.listdir(query_folder):
    if f.lower().endswith((".jpg", ".png", ".jpeg")):
        query_image = os.path.join(query_folder, f)
        break

query = preprocess(
    Image.open(query_image).convert("RGB")
).unsqueeze(0).to(device)

with torch.no_grad():
    q_emb = model.encode_image(query)

scores = cosine_similarity(q_emb.cpu(), torch.cat(db_emb).cpu())[0]
i = scores.argmax()

print("Query image:", os.path.basename(query_image))
print("Best match:", names[i])
print("Similarity:", scores[i]*100)
