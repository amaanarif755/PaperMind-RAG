v1 = [1,2,3,4]
v2 = [1,2,3,8]

def cosine_similarity(v1, v2):
    multiplication = sum(a*b for a, b in zip(v1, v2))
    v1_length = sum(a**2 for a in v1) ** 0.5
    v2_length = sum(b**2 for b in v2) ** 0.5
    if v1_length == 0 or v2_length == 0:
        return 0.0
    return multiplication / (v1_length * v2_length)

similarity = cosine_similarity(v1, v2)
print(f"Cosine Similarity: {similarity}")


v3=[1,2,3,4]
v4 = [-1,-2,-3,-4]
similarity = cosine_similarity(v3, v4)
print(f"Cosine Similarity: {similarity}")