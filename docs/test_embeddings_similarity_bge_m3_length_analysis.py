from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rookeen.analyzers.embeddings_backends import get_backend

pytestmark = [pytest.mark.slow]
DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "test_data" / "embeddings"


def _cosine(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def test_bge_m3_length_sensitivity_analysis() -> None:
    """Test BGE-M3 embeddings sensitivity to text length.
    
    Uses backend directly (not CLI) and tests 8 pairs with progressively
    longer sentences to investigate if similarity scores improve with length.
    """
    # Get initial pair from existing JSONL files
    similar_path = DATA_DIR / "pairs_similar.jsonl"
    dissimilar_path = DATA_DIR / "pairs_dissimilar.jsonl"
    
    initial_similar: dict[str, str] | None = None
    initial_dissimilar: dict[str, str] | None = None
    
    with similar_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                initial_similar = json.loads(line)
                break
    
    with dissimilar_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                initial_dissimilar = json.loads(line)
                break
    
    if not initial_similar or not initial_dissimilar:
        pytest.skip("Initial test pairs not found")
    
    # Hardcode 8 pairs with progressively longer sentences
    # Pair 1: Initial pair from JSONL files
    # mypy: narrow after None-check
    assert initial_similar is not None and initial_dissimilar is not None
    similar_pairs: list[tuple[str, str]] = [
        (initial_similar["text1"], initial_similar["text2"]),
    ]
    dissimilar_pairs: list[tuple[str, str]] = [
        (initial_dissimilar["text1"], initial_dissimilar["text2"]),
    ]
    
    # Pairs 2-8: Progressively longer sentences
    # Similar pairs - maintaining semantic similarity while increasing length
    similar_pairs.extend([
        # Pair 2: Slightly longer
        (
            "Dogs are friendly animals that make great companions.",
            "Dogs are very friendly pets that make excellent companions for families.",
        ),
        # Pair 3: Medium length
        (
            "Dogs are friendly animals that make great companions. They are loyal and caring creatures who bring joy to many households.",
            "Dogs are very friendly pets that make excellent companions for families. These loyal and caring animals bring happiness and joy to countless homes.",
        ),
        # Pair 4: Longer
        (
            "Dogs are friendly animals that make great companions. They are loyal and caring creatures who bring joy to many households. Many people find that owning a dog improves their quality of life significantly.",
            "Dogs are very friendly pets that make excellent companions for families. These loyal and caring animals bring happiness and joy to countless homes. Numerous studies show that dog ownership greatly enhances the overall quality of life for many individuals.",
        ),
        # Pair 5: Even longer
        (
            "Dogs are friendly animals that make great companions. They are loyal and caring creatures who bring joy to many households. Many people find that owning a dog improves their quality of life significantly. Dogs require regular exercise, which encourages their owners to stay active and healthy.",
            "Dogs are very friendly pets that make excellent companions for families. These loyal and caring animals bring happiness and joy to countless homes. Numerous studies show that dog ownership greatly enhances the overall quality of life for many individuals. Regular exercise needs of dogs motivate their owners to maintain an active and healthy lifestyle.",
        ),
        # Pair 6: Very long
        (
            "Dogs are friendly animals that make great companions. They are loyal and caring creatures who bring joy to many households. Many people find that owning a dog improves their quality of life significantly. Dogs require regular exercise, which encourages their owners to stay active and healthy. Additionally, dogs provide emotional support and can help reduce stress and anxiety in their human companions.",
            "Dogs are very friendly pets that make excellent companions for families. These loyal and caring animals bring happiness and joy to countless homes. Numerous studies show that dog ownership greatly enhances the overall quality of life for many individuals. Regular exercise needs of dogs motivate their owners to maintain an active and healthy lifestyle. Furthermore, dogs offer significant emotional support and are known to help decrease stress levels and anxiety among their human family members.",
        ),
        # Pair 7: Extra long
        (
            "Dogs are friendly animals that make great companions. They are loyal and caring creatures who bring joy to many households. Many people find that owning a dog improves their quality of life significantly. Dogs require regular exercise, which encourages their owners to stay active and healthy. Additionally, dogs provide emotional support and can help reduce stress and anxiety in their human companions. Training a dog also teaches responsibility and patience, valuable life skills that benefit both children and adults.",
            "Dogs are very friendly pets that make excellent companions for families. These loyal and caring animals bring happiness and joy to countless homes. Numerous studies show that dog ownership greatly enhances the overall quality of life for many individuals. Regular exercise needs of dogs motivate their owners to maintain an active and healthy lifestyle. Furthermore, dogs offer significant emotional support and are known to help decrease stress levels and anxiety among their human family members. The process of training a dog additionally instills important values like responsibility and patience, which are beneficial life skills for people of all ages.",
        ),
        # Pair 8: Extremely long
        (
            "Dogs are friendly animals that make great companions. They are loyal and caring creatures who bring joy to many households. Many people find that owning a dog improves their quality of life significantly. Dogs require regular exercise, which encourages their owners to stay active and healthy. Additionally, dogs provide emotional support and can help reduce stress and anxiety in their human companions. Training a dog also teaches responsibility and patience, valuable life skills that benefit both children and adults. The bond between a dog and its owner is often described as one of the strongest relationships in the animal kingdom, built on trust, mutual respect, and unconditional love.",
            "Dogs are very friendly pets that make excellent companions for families. These loyal and caring animals bring happiness and joy to countless homes. Numerous studies show that dog ownership greatly enhances the overall quality of life for many individuals. Regular exercise needs of dogs motivate their owners to maintain an active and healthy lifestyle. Furthermore, dogs offer significant emotional support and are known to help decrease stress levels and anxiety among their human family members. The process of training a dog additionally instills important values like responsibility and patience, which are beneficial life skills for people of all ages. The connection between a dog and its human is frequently characterized as among the most profound relationships in the entire animal world, founded upon principles of trust, mutual respect, and unconditional affection.",
        ),
    ])
    
    # Dissimilar pairs - maintaining semantic dissimilarity while increasing length
    dissimilar_pairs.extend([
        # Pair 2: Slightly longer
        (
            "The sun is bright today and the sky is clear.",
            "I cooked pasta for dinner last night and it was delicious.",
        ),
        # Pair 3: Medium length
        (
            "The sun is bright today and the sky is clear. It's a perfect day for outdoor activities and enjoying nature.",
            "I cooked pasta for dinner last night and it was delicious. The recipe included fresh tomatoes, garlic, and basil.",
        ),
        # Pair 4: Longer
        (
            "The sun is bright today and the sky is clear. It's a perfect day for outdoor activities and enjoying nature. Many people are taking advantage of the beautiful weather to go hiking or have picnics in the park.",
            "I cooked pasta for dinner last night and it was delicious. The recipe included fresh tomatoes, garlic, and basil. I learned this recipe from my grandmother, who taught me many traditional Italian cooking techniques.",
        ),
        # Pair 5: Even longer
        (
            "The sun is bright today and the sky is clear. It's a perfect day for outdoor activities and enjoying nature. Many people are taking advantage of the beautiful weather to go hiking or have picnics in the park. The temperature is just right, not too hot and not too cold, making it ideal for spending time outside.",
            "I cooked pasta for dinner last night and it was delicious. The recipe included fresh tomatoes, garlic, and basil. I learned this recipe from my grandmother, who taught me many traditional Italian cooking techniques. She showed me how to properly prepare the sauce by simmering it slowly to develop rich flavors.",
        ),
        # Pair 6: Very long
        (
            "The sun is bright today and the sky is clear. It's a perfect day for outdoor activities and enjoying nature. Many people are taking advantage of the beautiful weather to go hiking or have picnics in the park. The temperature is just right, not too hot and not too cold, making it ideal for spending time outside. Birds are chirping in the trees, and the gentle breeze makes the day even more pleasant.",
            "I cooked pasta for dinner last night and it was delicious. The recipe included fresh tomatoes, garlic, and basil. I learned this recipe from my grandmother, who taught me many traditional Italian cooking techniques. She showed me how to properly prepare the sauce by simmering it slowly to develop rich flavors. The final dish was served with grated parmesan cheese and a drizzle of olive oil, just as she always recommended.",
        ),
        # Pair 7: Extra long
        (
            "The sun is bright today and the sky is clear. It's a perfect day for outdoor activities and enjoying nature. Many people are taking advantage of the beautiful weather to go hiking or have picnics in the park. The temperature is just right, not too hot and not too cold, making it ideal for spending time outside. Birds are chirping in the trees, and the gentle breeze makes the day even more pleasant. Children are playing in the playground, and families are setting up blankets for afternoon gatherings.",
            "I cooked pasta for dinner last night and it was delicious. The recipe included fresh tomatoes, garlic, and basil. I learned this recipe from my grandmother, who taught me many traditional Italian cooking techniques. She showed me how to properly prepare the sauce by simmering it slowly to develop rich flavors. The final dish was served with grated parmesan cheese and a drizzle of olive oil, just as she always recommended. Cooking has become one of my favorite hobbies, and I enjoy experimenting with different flavors and ingredients.",
        ),
        # Pair 8: Extremely long
        (
            "The sun is bright today and the sky is clear. It's a perfect day for outdoor activities and enjoying nature. Many people are taking advantage of the beautiful weather to go hiking or have picnics in the park. The temperature is just right, not too hot and not too cold, making it ideal for spending time outside. Birds are chirping in the trees, and the gentle breeze makes the day even more pleasant. Children are playing in the playground, and families are setting up blankets for afternoon gatherings. Weather like this reminds us why we appreciate the natural world and the simple pleasures it provides.",
            "I cooked pasta for dinner last night and it was delicious. The recipe included fresh tomatoes, garlic, and basil. I learned this recipe from my grandmother, who taught me many traditional Italian cooking techniques. She showed me how to properly prepare the sauce by simmering it slowly to develop rich flavors. The final dish was served with grated parmesan cheese and a drizzle of olive oil, just as she always recommended. Cooking has become one of my favorite hobbies, and I enjoy experimenting with different flavors and ingredients. These culinary experiences connect me to my heritage and help me create meaningful memories in the kitchen.",
        ),
    ])
    
    # Initialize backend
    backend = get_backend("bge-m3", model_name="BAAI/bge-m3")
    backend.load()
    
    # Test similar pairs
    print("\n" + "=" * 80)
    print("SIMILAR PAIRS - Similarity Scores")
    print("=" * 80)
    similar_scores = []
    for i, (text1, text2) in enumerate(similar_pairs, 1):
        v1 = backend.embed(text1)
        v2 = backend.embed(text2)
        cos = _cosine(v1, v2)
        similar_scores.append(cos)
        len1, len2 = len(text1), len(text2)
        print(f"\nPair {i}:")
        print(f"  Text1 length: {len1} chars")
        print(f"  Text2 length: {len2} chars")
        print(f"  Cosine similarity: {cos:.6f}")
        print(f"  Threshold: >= 0.65 | {'PASS' if cos >= 0.65 else 'FAIL'}")
        print(f"  Text1: {text1[:100]}{'...' if len(text1) > 100 else ''}")
        print(f"  Text2: {text2[:100]}{'...' if len(text2) > 100 else ''}")
    
    # Test dissimilar pairs
    print("\n" + "=" * 80)
    print("DISSIMILAR PAIRS - Similarity Scores")
    print("=" * 80)
    dissimilar_scores = []
    for i, (text1, text2) in enumerate(dissimilar_pairs, 1):
        v1 = backend.embed(text1)
        v2 = backend.embed(text2)
        cos = _cosine(v1, v2)
        dissimilar_scores.append(cos)
        len1, len2 = len(text1), len(text2)
        print(f"\nPair {i}:")
        print(f"  Text1 length: {len1} chars")
        print(f"  Text2 length: {len2} chars")
        print(f"  Cosine similarity: {cos:.6f}")
        print(f"  Threshold: <= 0.65 | {'PASS' if cos <= 0.65 else 'FAIL'}")
        print(f"  Text1: {text1[:100]}{'...' if len(text1) > 100 else ''}")
        print(f"  Text2: {text2[:100]}{'...' if len(text2) > 100 else ''}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nSimilar pairs scores:")
    for i, score in enumerate(similar_scores, 1):
        print(f"  Pair {i}: {score:.6f} {'PASS' if score >= 0.65 else 'FAIL'}")
    
    print("\nDissimilar pairs scores:")
    for i, score in enumerate(dissimilar_scores, 1):
        print(f"  Pair {i}: {score:.6f} {'PASS' if score <= 0.65 else 'FAIL'}")
    
    print("\nLength progression analysis:")
    print("  Similar pairs - scores increasing with length:")
    print(f"    First: {similar_scores[0]:.6f}, Last: {similar_scores[-1]:.6f}, "
          f"Change: {similar_scores[-1] - similar_scores[0]:+.6f}")
    print("  Dissimilar pairs - scores staying low:")
    print(f"    First: {dissimilar_scores[0]:.6f}, Last: {dissimilar_scores[-1]:.6f}, "
          f"Change: {dissimilar_scores[-1] - dissimilar_scores[0]:+.6f}")
    
    # Assert thresholds (will fail if hypothesis is wrong for first pairs)
    for i, (text1, text2) in enumerate(similar_pairs, 1):
        v1 = backend.embed(text1)
        v2 = backend.embed(text2)
        cos = _cosine(v1, v2)
        assert cos >= 0.65, f"Similar pair {i} failed threshold: {cos:.6f} < 0.65"
    
    for i, (text1, text2) in enumerate(dissimilar_pairs, 1):
        v1 = backend.embed(text1)
        v2 = backend.embed(text2)
        cos = _cosine(v1, v2)
        assert cos <= 0.65, f"Dissimilar pair {i} failed threshold: {cos:.6f} > 0.65"


