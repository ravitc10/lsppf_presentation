import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ========================================
# LOCAL CONFIG
# ========================================
INPUT_FILE = Path("final_1.json")
OUTPUT_FILE = Path("final_1_with_embeddings.json")

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64
MAX_LENGTH = 256


# ========================================
# Text helper
# ========================================
def build_text(entry: dict) -> str:
    """
    Adapted for comments JSON:
      - Name
      - Comment

    Returns: "Name: Comment"
    """
    name = (entry.get("Name") or "").strip()
    comment = (entry.get("Comment") or "").strip()

    if name and comment:
        return f"{name}: {comment}"
    if comment:
        return comment
    return name


# ========================================
# SBERT SETUP (mean pooling)
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(SBERT_MODEL_NAME)
base_model = AutoModel.from_pretrained(SBERT_MODEL_NAME).to(device)
base_model.eval()


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * input_mask_expanded).sum(dim=1)
    counts = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def sbert_generate_embeddings_batch(texts):
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        outputs = base_model(**encoded)
        pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])

        # L2 normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        all_embeddings.append(pooled.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings.tolist()


# ========================================
# MAIN
# ========================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing {INPUT_FILE.resolve()}"
    )

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError("Expected a list of dicts in comments_1.json")

texts = []
valid_indices = []

for idx, entry in enumerate(data):
    text = build_text(entry).strip()
    if text:
        texts.append(text)
        valid_indices.append(idx)

print(f"Loaded entries: {len(data)}")
print(f"Embedding texts: {len(texts)}")

if texts:
    embeddings = sbert_generate_embeddings_batch(texts)

    for emb_idx, data_idx in enumerate(tqdm(valid_indices, desc="Attaching embeddings")):
        emb = embeddings[emb_idx]
        data[data_idx]["embedding"] = [round(x, 8) for x in emb]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

with_embeddings = sum(1 for d in data if "embedding" in d)
print(f"\nSaved: {OUTPUT_FILE.resolve()}")
print(f"Entries with embeddings: {with_embeddings}/{len(data)}")

# Preview
for d in data:
    if "embedding" in d:
        print("\nSample:")
        print(d.get("Name"))
        print(d.get("Comment")[:100], "...")
        print("Embedding length:", len(d["embedding"]))
        break
