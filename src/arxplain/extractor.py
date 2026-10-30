"""Concept extraction across the dimensions ArXplain adapts to.

Heuristic-first (known vocabularies + patterns) so it works offline;
an LLM, when configured, only enriches descriptions.
"""
from __future__ import annotations

import re

from .models import Concept, Dimension, Paper

# Implementation mentions are NOT topics: a paper's strategy/algorithm
# is independent of the language or framework one team happened to use.
# They are collected only to be demoted into a single setup note.
IMPLEMENTATION_TERMS = ["PyTorch", "TensorFlow", "JAX", "Keras",
                        "scikit-learn", "Hugging Face", "TensorFlow",
                        "Python", "C++", "Rust", "Julia", "CUDA", "Docker",
                        "Kubernetes", "Weights & Biases", "W&B", "MLflow",
                        "TensorBoard", "Git", "Linux", "ONNX", "Ray"]
DATASETS = ["MNIST", "CIFAR-10", "CIFAR-100", "ImageNet", "COCO", "GLUE",
            "SQuAD", "WikiText", "LibriSpeech", "MMLU", "HumanEval", "MS MARCO"]
METRICS = ["accuracy", "precision", "recall", "F1", "BLEU", "ROUGE", "perplexity",
           "mAP", "AUC", "RMSE", "latency", "throughput", "FID"]

MATH_PATTERNS: list[tuple[str, str, int]] = [
    (r"softmax|attention mechanism|scaled dot-product", "Scaled dot-product attention & softmax", 2),
    (r"gradient descent|backpropagation|stochastic gradient", "Gradient descent & backpropagation", 2),
    (r"cross[- ]entropy|loss function", "Loss functions & cross-entropy", 2),
    (r"linear algebra|matrix multiplication|eigen", "Linear algebra (matrices & vectors)", 1),
    (r"probabilit|bayesian|distribution", "Probability & distributions", 2),
    (r"theorem|proof|lemma", "Proofs & formal reasoning", 3),
    (r"optimization|convex|lagrang", "Optimization theory", 3),
    (r"fourier|laplace transform", "Integral transforms (Fourier/Laplace)", 3),
    (r"\$[^$]+\$|\\sum|\\frac|equation\s*\(?\d", "Mathematical notation & equations", 1),
]
THEORY_PATTERNS: list[tuple[str, str, int]] = [
    (r"transformer|self-attention", "Transformer architecture", 2),
    (r"convolutional|CNN", "Convolutional neural networks", 1),
    (r"recurrent|LSTM|GRU", "Recurrent networks (LSTM/GRU)", 2),
    (r"reinforcement learning|policy gradient|Q-learning", "Reinforcement learning", 3),
    (r"diffusion model|denoising", "Diffusion models", 3),
    (r"embedding|representation learning", "Embeddings & representation learning", 1),
    (r"regularization|dropout|weight decay", "Regularization (dropout, weight decay)", 1),
    (r"normalization|batch norm|layer norm", "Normalization layers", 1),
]

RESULTS_PATTERNS: list[tuple[str, str, int, str]] = [
    (r"outperform|state-of-the-art|achieves?\s+\d|superior quality|improv\w+ over",
     "Main results & headline findings", 2,
     "What the experiments actually showed, in numbers."),
    (r"we observe|results show|we find that|our results|finding",
     "Interpreting the results", 2,
     "What the numbers mean — and what they do not show."),
    (r"ablation|removing|without the|variant",
     "Ablation results: what each part contributes", 2,
     "Evidence that each component of the method earns its place."),
]
ASSUMPTION_PATTERNS: list[tuple[str, str, int, str]] = [
    (r"we assume|assumption|assumes? that|assuming",
     "Assumptions the method makes", 2,
     "Things the paper takes for granted; the method may break without them."),
    (r"i\.i\.d\.|independent and identically|independently distributed",
     "Data assumptions (i.i.d. and similar)", 3,
     "Statistical assumptions about where the data comes from."),
    (r"for simplicity|simplifying|without loss of generality|we consider the case",
     "Simplifying assumptions & scope choices", 2,
     "Cases the paper deliberately restricts itself to."),
]
LIMITATION_PATTERNS: list[tuple[str, str, int, str]] = [
    (r"limitation|does not handle|fails? to|failure case|cannot handle",
     "Limitations & failure cases", 2,
     "Where the approach breaks or was not tested."),
    (r"quadratic|complexity of|computational cost|memory requirement|scales? (poorly|with)",
     "Complexity & scaling constraints", 2,
     "How cost grows with input size — often the real constraint."),
    (r"future work|remains? to be|open problem|beyond the scope",
     "Open problems & future work", 1,
     "What the authors themselves leave undone."),
    (r"large amounts of data|data-hungry|expensive to train|training cost",
     "Data & compute constraints", 2,
     "What it costs, in data and compute, to make this work."),
]

def _evidence(text: str, pattern: str) -> str:
    m = re.search(pattern, text, re.I)
    if not m:
        return ""
    s = max(0, m.start() - 70)
    return re.sub(r"\s+", " ", text[s:m.end() + 90]).strip()


def extract_concepts(paper: Paper) -> list[Concept]:
    text = paper.text
    concepts: list[Concept] = []
    seen: set[tuple[str, str]] = set()

    def add(c: Concept) -> None:
        key = (c.dimension.value, c.name.lower())
        if key not in seen:
            seen.add(key)
            concepts.append(c)

    for pattern, name, lvl in MATH_PATTERNS:
        if re.search(pattern, text, re.I):
            add(Concept(name=name, dimension=Dimension.MATHEMATICS,
                        required_level=lvl, evidence=_evidence(text, pattern),
                        description=f"Mathematics the paper relies on: {name.lower()}."))
    for pattern, name, lvl in THEORY_PATTERNS:
        if re.search(pattern, text, re.I):
            add(Concept(name=name, dimension=Dimension.THEORY, required_level=lvl,
                        evidence=_evidence(text, pattern),
                        description=f"Core idea used by the paper: {name.lower()}."))
    mentioned = [term for term in IMPLEMENTATION_TERMS
                 if re.search(rf"\b{re.escape(term)}\b", text, re.I)]
    if mentioned:
        add(Concept(
            name="Experimental setup (implementation details)",
            dimension=Dimension.EXPERIMENTATION, required_level=1,
            evidence=_evidence(text, re.escape(mentioned[0])),
            description=("Mentioned only as how this team implemented the "
                         "work (" + ", ".join(mentioned[:5]) + "). The "
                         "paper's ideas do not depend on these choices; "
                         "they matter only for reproducing the experiments.")))
    for patterns, dim in ((RESULTS_PATTERNS, Dimension.RESULTS),
                          (ASSUMPTION_PATTERNS, Dimension.ASSUMPTIONS),
                          (LIMITATION_PATTERNS, Dimension.LIMITATIONS)):
        for pattern, name, lvl, desc in patterns:
            if re.search(pattern, text, re.I):
                add(Concept(name=name, dimension=dim, required_level=lvl,
                            evidence=_evidence(text, pattern), description=desc))
    for term in DATASETS:
        if re.search(rf"\b{re.escape(term)}\b", text):
            add(Concept(name=f"Dataset: {term}", dimension=Dimension.EXPERIMENTATION,
                        required_level=1, evidence=_evidence(text, re.escape(term)),
                        description=f"Experiments are run on {term}."))
    found_metrics = [m for m in METRICS if re.search(rf"\b{m}\b", text, re.I)]
    if found_metrics:
        add(Concept(name="Evaluation metrics: " + ", ".join(found_metrics[:4]),
                    dimension=Dimension.EXPERIMENTATION, required_level=1,
                    evidence=_evidence(text, re.escape(found_metrics[0])),
                    description="How the paper measures success."))
    if re.search(r"baseline|ablation|state-of-the-art|SOTA", text, re.I):
        add(Concept(name="Baselines & ablation studies",
                    dimension=Dimension.EXPERIMENTATION, required_level=2,
                    evidence=_evidence(text, r"baseline|ablation"),
                    description="Comparisons that show each part actually helps."))
    if re.search(r"hyperparameter|learning rate|batch size|epochs", text, re.I):
        add(Concept(name="Hyperparameters & training setup",
                    dimension=Dimension.EXPERIMENTATION, required_level=1,
                    evidence=_evidence(text, r"hyperparameter|learning rate"),
                    description="Knobs that control training."))
    # Keywords: frequent capitalised / quoted terms the paper defines.
    for term in sorted(set(re.findall(r'"([A-Z][A-Za-z\- ]{3,30})"', text)))[:6]:
        add(Concept(name=term, dimension=Dimension.KEYWORDS, required_level=1,
                    evidence=_evidence(text, re.escape(term)),
                    description=f"Term defined or emphasised by the paper: {term}."))
    paper.concepts = concepts
    return concepts
