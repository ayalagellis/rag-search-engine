def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors must have the same length, got {len(vec1)} and {len(vec2)}"
        )
    return [vec1[i] + vec2[i] for i in range(len(vec1))]


def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors must have the same length, got {len(vec1)} and {len(vec2)}"
        )
    return [vec1[i] - vec2[i] for i in range(len(vec1))]


def dot(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors must have the same length, got {len(vec1)} and {len(vec2)}"
        )
    total = 0.0
    for i in range(len(vec1)):
        total += vec1[i] * vec2[i]
    return total


def euclidean_norm(vec: list[float]) -> float:
    return sum(x ** 2 for x in vec) ** 0.5


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors must have the same length, got {len(vec1)} and {len(vec2)}"
        )
    dot_product = dot(vec1, vec2)
    mag1 = euclidean_norm(vec1)
    mag2 = euclidean_norm(vec2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)