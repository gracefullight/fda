import math

import matplotlib.pyplot as plt

# Points
A = (1, 8)
B = (7, 1)

# --- Calculations ---
dx = B[0] - A[0]
dy = B[1] - A[1]

manhattan = abs(dx) + abs(dy)
euclidean = math.hypot(dx, dy)

dot = A[0] * B[0] + A[1] * B[1]
normA = math.hypot(A[0], A[1])
normB = math.hypot(B[0], B[1])
cosine_similarity = dot / (normA * normB)

print(f"Manhattan distance: {manhattan}")
print(f"Euclidean distance: {euclidean:.6f}")
print(f"Cosine similarity:  {cosine_similarity:.6f}")


# --- Helper for consistent axes ---
def setup_axes(title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.grid(b=True)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("x")
    plt.ylabel("y")


# --- 1) Manhattan path (axis-aligned) ---
setup_axes("Manhattan Path: A → B in 10x10 Space")
plt.scatter(*A, s=80, label="A (1,8)")
plt.scatter(*B, s=80, label="B (7,1)")
# One L-shaped path: horizontal then vertical
plt.plot([A[0], B[0]], [A[1], A[1]], linestyle="--")
plt.plot([B[0], B[0]], [A[1], B[1]], linestyle="--")
plt.legend()
plt.tight_layout()

# --- 2) Euclidean distance (straight line) ---
setup_axes("Euclidean Distance: A ↔ B in 10x10 Space")
plt.scatter(*A, s=80, label="A (1,8)")
plt.scatter(*B, s=80, label="B (7,1)")
plt.plot([A[0], B[0]], [A[1], B[1]], linewidth=2, label="Euclidean Line")
plt.legend()
plt.tight_layout()

# --- 3) Cosine similarity (vectors from origin) ---
setup_axes(f"Cosine Similarity Vectors (cos θ = {cosine_similarity:.3f})")
origin = (0, 0)
plt.quiver(*origin, *A, angles="xy", scale_units="xy", scale=1, label="Vector A")
plt.quiver(*origin, *B, angles="xy", scale_units="xy", scale=1, label="Vector B")
plt.scatter(*A, s=50)
plt.scatter(*B, s=50)
plt.text(A[0] + 0.2, A[1], "A", fontsize=10)
plt.text(B[0] + 0.2, B[1], "B", fontsize=10)
plt.legend(loc="upper left")
plt.tight_layout()

plt.show()
