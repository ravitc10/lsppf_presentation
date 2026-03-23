import json
import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial import cKDTree
from tqdm import tqdm


# =========================
# LOCAL CONFIG
# =========================
INPUT_PATH = "final_1_with_embeddings.json"   # <-- IMPORTANT: use embedded file
OUTPUT_FRONTEND = "final_1_tsne.json"
OUTPUT_BACKEND = ""

PERPLEXITY = 30.0
MIN_DIST = 1.0
MAGNIFICATION = 2.0
JITTER_RADIUS = 0.01
RANDOM_SEED = 42


# =========================
# Overlap separation (unchanged)
# =========================
def separate_overlapping_points(coords, similarity_matrix, keys,
                                min_dist=1.0, magnification=2.0, jitter_radius=0.01, random_seed=42):

    n = coords.shape[0]
    print(f"Magnifying coordinates by factor {magnification}...")

    coords_scaled = coords * magnification

    print("Finding exact duplicates...")
    tree = cKDTree(coords_scaled)
    duplicate_threshold = min_dist * magnification * 0.1
    pairs = tree.query_pairs(duplicate_threshold, output_type='set')

    keep_together = set()

    print(f"Applying jitter to {len(pairs)} duplicate pairs...")
    np.random.seed(random_seed)
    jittered = np.zeros(n, dtype=bool)

    for i, j in pairs:
        if not jittered[i] and not jittered[j]:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, jitter_radius * magnification)
            jitter = radius * np.array([np.cos(angle), np.sin(angle)])

            coords_scaled[i] += jitter
            coords_scaled[j] -= jitter
            jittered[i] = True
            jittered[j] = True

    print("Handling remaining exact coordinate matches...")
    step = jitter_radius * magnification * 0.1
    if step <= 0:
        step = 1e-6

    rounded_coords = np.round(coords_scaled / step) * step
    unique_coords, inverse, counts = np.unique(
        rounded_coords, axis=0, return_inverse=True, return_counts=True
    )

    for idx in np.where(counts > 1)[0]:
        group_indices = np.where(inverse == idx)[0]
        n_in_group = len(group_indices)
        if n_in_group > 1:
            center = coords_scaled[group_indices[0]].copy()
            angles = np.linspace(0, 2 * np.pi, n_in_group, endpoint=False)
            for k, gi in enumerate(group_indices):
                angle = angles[k]
                offset = jitter_radius * magnification * np.array(
                    [np.cos(angle), np.sin(angle)]
                )
                coords_scaled[gi] = center + offset

    return coords_scaled


# =========================
# Load data
# =========================
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

entries = []
for d in data:
    emb = d.get("embedding")
    if isinstance(emb, list) and len(emb) > 0:
        entries.append(d)

if not entries:
    raise RuntimeError("No embeddings found. Run embedding script first.")

# =========================
# Labels for comments
# =========================
names = [d.get("Name", "").strip() for d in entries]
comments = [d.get("Comment", "").strip() for d in entries]

labels = [
    f"{n}: {c[:120]}..." if len(c) > 120 else f"{n}: {c}"
    for n, c in zip(names, comments)
]

X = np.array([d["embedding"] for d in entries], dtype=np.float32)

n, dim = X.shape
print(f"Loaded {n} comments with embeddings (dim={dim}).")

# Normalize
norms = np.linalg.norm(X, axis=1, keepdims=True)
norms[norms == 0] = 1.0
X = X / norms


# =========================
# Similarity + distance
# =========================
print("Computing cosine similarity...")
sim_matrix = X @ X.T
sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
np.fill_diagonal(sim_matrix, 1.0)

print("Converting to distance...")
dist_matrix = 1.0 - sim_matrix
np.fill_diagonal(dist_matrix, 0.0)
dist_matrix = (dist_matrix + dist_matrix.T) / 2.0


# =========================
# t-SNE
# =========================
if n < 3:
    raise RuntimeError("Need at least 3 points for t-SNE.")

safe_perplexity = min(PERPLEXITY, max(2.0, (n - 1) / 3.0))

print(f"Running t-SNE (perplexity={safe_perplexity:.2f})...")
tsne = TSNE(
    n_components=2,
    metric="precomputed",
    random_state=RANDOM_SEED,
    perplexity=safe_perplexity,
    init="random",
)

coords = tsne.fit_transform(dist_matrix)

coords = separate_overlapping_points(
    coords,
    sim_matrix,
    keys=[("comment", i) for i in range(n)],
    min_dist=MIN_DIST,
    magnification=MAGNIFICATION,
    jitter_radius=JITTER_RADIUS,
    random_seed=RANDOM_SEED,
)


# =========================
# Save output
# =========================
output = []
for i in range(n):
    x, y = coords[i]
    output.append({
        "name": names[i],
        "comment": comments[i],
        "label": labels[i],
        "x": float(x),
        "y": float(y),
    })

with open(OUTPUT_FRONTEND, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

if OUTPUT_BACKEND:
    with open(OUTPUT_BACKEND, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

print(f"Saved t-SNE map to {OUTPUT_FRONTEND}")
