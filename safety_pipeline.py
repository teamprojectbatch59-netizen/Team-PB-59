"""Local prompt-injection screening and Ollama Llama Guard categorization.

Stage 1 uses the mean of each detector's strongest overlapping-window score.
These are model scores; tuning a decision threshold does not calibrate them.
Inference never downloads checkpoints. Install/cache the models explicitly.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import threading

import requests


APP_DIR = Path(__file__).resolve().parent
STAGE1_MODEL_IDS = {
    "prompt_guard": "meta-llama/Llama-Prompt-Guard-2-86M",
    "deberta": "protectai/deberta-v3-base-prompt-injection-v2",
}
_MODEL_ENV = {
    "prompt_guard": "RUAS_PROMPT_GUARD_MODEL",
    "deberta": "RUAS_DEBERTA_INJECTION_MODEL",
}
_MODEL_NAMES = {"prompt_guard": "Prompt Guard 2", "deberta": "DeBERTa-v3 injection"}
AGGREGATION = "max_windows_then_mean"
MAX_SEQUENCE_LENGTH = 512
WINDOW_OVERLAP = 64
_MODEL_CACHE: dict = {}
_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_TORCH_CONFIGURED = False

# Meta's Llama Guard 3-1B taxonomy (not the ModernBERT tactic label space).
LLAMA_GUARD_CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
}


class SafetyPipelineUnavailable(RuntimeError):
    """Required screening or categorization could not produce a valid result."""


def _local_path(value: str) -> Path:
    """Resolve a path that may be absolute, relative to app dir, or a Hugging Face repo ID."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # Relative paths are resolved relative to the app directory
    return APP_DIR / path


def _looks_like_local_path(value: str) -> bool:
    """Check if a string looks like a local filesystem path (not a HF repo ID)."""
    return (
        Path(value).is_absolute()
        or value.startswith((".", "models/", "models\\"))
        or ("/" in value and not value.startswith(("http://", "https://")) and not value.count("/") == 1)
    )


def stage1_model_ids() -> dict[str, str]:
    """Return configured checkpoint identities, resolving local paths consistently."""
    sources = {}
    for key, default in STAGE1_MODEL_IDS.items():
        value = os.environ.get(_MODEL_ENV[key], default).strip()
        if not value:
            raise SafetyPipelineUnavailable(f"{_MODEL_ENV[key]} must name a model or local directory.")
        path = _local_path(value)
        # If it looks like a local path or the resolved path exists, use the resolved absolute path
        if _looks_like_local_path(value) or path.exists():
            value = str(path.resolve())
        sources[key] = value
    return sources


def _probability(value, name: str) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SafetyPipelineUnavailable(f"{name} must be a finite number in [0, 1].") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise SafetyPipelineUnavailable(f"{name} must be a finite number in [0, 1].")
    return number


def get_stage1_threshold() -> tuple[float, str]:
    """Read an explicit override or a threshold fitted on validation data.

The untuned default is exposed as such. A malformed or incompatible saved
calibration is an error, rather than silently changing the operating point.
"""
    if "RUAS_STAGE1_THRESHOLD" in os.environ:
        return _probability(os.environ["RUAS_STAGE1_THRESHOLD"], "RUAS_STAGE1_THRESHOLD"), "environment_override"
    configured = os.environ.get("RUAS_STAGE1_CALIBRATION", "training/results/stage1_calibration.json")
    if not configured.strip():
        raise SafetyPipelineUnavailable("RUAS_STAGE1_CALIBRATION cannot be empty.")
    path = _local_path(configured)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        if "RUAS_STAGE1_CALIBRATION" in os.environ:
            raise SafetyPipelineUnavailable(f"Configured Stage 1 calibration was not found: {path}.") from exc
        return 0.5, "default_untuned"
    except (OSError, ValueError) as exc:
        raise SafetyPipelineUnavailable(f"Cannot read Stage 1 calibration: {path}.") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise SafetyPipelineUnavailable("Stage 1 calibration requires schema_version 1.")
    if data.get("models") != stage1_model_ids() or data.get("aggregation") != AGGREGATION:
        raise SafetyPipelineUnavailable("Stage 1 calibration does not match the configured checkpoints and aggregation. Recalibrate.")
    return _probability(data.get("threshold"), "calibration threshold"), "validation_calibration"


def _positive_label_index(config, key: str) -> int:
    labels = {int(index): str(label).upper() for index, label in (config.id2label or {}).items()}
    if int(config.num_labels) != 2:
        raise SafetyPipelineUnavailable(f"{_MODEL_NAMES[key]} must have two output classes.")
    expected = "MALICIOUS" if key == "prompt_guard" else "INJECTION"
    matches = [index for index, label in labels.items() if label == expected]
    if len(matches) == 1:
        return matches[0]
    # The original Prompt Guard 2 config used generic labels. Meta's official
    # inference utility explicitly selects probability column 1 for this model:
    # github.com/meta-llama/llama-cookbook/.../prompt_guard/inference.py
    if key == "prompt_guard" and labels in ({}, {0: "LABEL_0", 1: "LABEL_1"}):
        return 1
    raise SafetyPipelineUnavailable(f"{_MODEL_NAMES[key]} has no recognized {expected} output label.")


def _load_classifier(key: str):
    global _TORCH_CONFIGURED
    source = stage1_model_ids()[key]
    requested_device = os.environ.get("RUAS_STAGE1_DEVICE", "cpu").strip().lower()
    if not re.fullmatch(r"cpu|auto|cuda(?::\d+)?", requested_device):
        raise SafetyPipelineUnavailable("RUAS_STAGE1_DEVICE must be cpu, auto, cuda, or cuda:<index>.")
    cache_key = (key, source, requested_device)
    with _LOAD_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if not _TORCH_CONFIGURED:
                threads = int(os.environ.get("RUAS_TORCH_THREADS", min(4, os.cpu_count() or 1)))
                torch.set_num_threads(max(1, min(16, threads)))
                _TORCH_CONFIGURED = True
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
            model = AutoModelForSequenceClassification.from_pretrained(
                source, local_files_only=True, trust_remote_code=False
            )
            positive_index = _positive_label_index(model.config, key)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ) if requested_device == "auto" else torch.device(requested_device)
            model.to(device)
            model.eval()
            loaded = model, tokenizer, torch, device, positive_index
        except Exception as exc:
            raise SafetyPipelineUnavailable(
                f"{_MODEL_NAMES[key]} is unavailable locally ({type(exc).__name__}). "
                f"Install/cache '{source}' or set {_MODEL_ENV[key]} to its local checkpoint directory. "
                "Prompt Guard 2 may require Hugging Face access approval. "
                f"Configured inference device: {requested_device}."
            ) from exc
        _MODEL_CACHE[cache_key] = loaded
        return loaded


def _tokenizer_input_ids_with_special_tokens(tokenizer, token_ids: list[int]) -> list[int]:
    """Add the model's boundary tokens across Transformers tokenizer versions."""
    builder = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(builder):
        return builder(token_ids)

    start_id = getattr(tokenizer, "cls_token_id", None)
    if start_id is None:
        start_id = getattr(tokenizer, "bos_token_id", None)
    end_id = getattr(tokenizer, "sep_token_id", None)
    if end_id is None:
        end_id = getattr(tokenizer, "eos_token_id", None)
    if start_id is None or end_id is None:
        raise SafetyPipelineUnavailable("Tokenizer does not expose supported start/end special-token IDs.")
    return [int(start_id), *[int(token_id) for token_id in token_ids], int(end_id)]


def _token_windows(tokenizer, text: str, max_length: int = MAX_SEQUENCE_LENGTH) -> list[list[int]]:
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    content_limit = max_length - special_count
    if content_limit <= WINDOW_OVERLAP:
        raise SafetyPipelineUnavailable("The detector tokenizer has an invalid sequence capacity.")
    step = content_limit - WINDOW_OVERLAP
    chunks = []
    for start in range(0, max(1, len(tokens)), step):
        content = tokens[start:start + content_limit]
        sequence = _tokenizer_input_ids_with_special_tokens(tokenizer, content)
        if len(sequence) > max_length:
            raise SafetyPipelineUnavailable("Detector tokenizer exceeded the window length.")
        chunks.append(sequence)
        if start + content_limit >= len(tokens):
            break
    return chunks


def _detector_result(key: str, probability: float, windows_scanned: int, device) -> dict:
    return {
        "name": _MODEL_NAMES[key],
        "model": stage1_model_ids()[key],
        "label": "Unsafe" if probability >= 0.5 else "Safe",
        "unsafe_probability": probability,
        "confidence": probability if probability >= 0.5 else 1 - probability,
        "threshold": 0.5,
        "windows_scanned": windows_scanned,
        "device": str(device),
    }


def _score_detector_batch(texts: list[str], key: str, batch_size: int | None = None) -> list[dict]:
    """Batch windows from different texts while retaining per-prompt maxima."""
    if not texts:
        return []
    model, tokenizer, torch, device, positive_index = _load_classifier(key)
    if tokenizer.pad_token_id is None:
        raise SafetyPipelineUnavailable(f"{_MODEL_NAMES[key]} tokenizer has no padding token.")
    try:
        size = int(batch_size if batch_size is not None else os.environ.get(
            "RUAS_STAGE1_BATCH_SIZE", 16 if str(device).startswith("cuda") else 4
        ))
    except (ValueError, TypeError) as exc:
        raise SafetyPipelineUnavailable("Stage 1 batch size must be an integer from 1 to 256.") from exc
    if not 1 <= size <= 256:
        raise SafetyPipelineUnavailable("Stage 1 batch size must be an integer from 1 to 256.")
    scores = [0.0] * len(texts)
    window_counts = [0] * len(texts)
    pending: list[list[int]] = []
    owners: list[int] = []

    def infer_pending():
        width = max(map(len, pending))
        input_ids = torch.full((len(pending), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(pending), width), dtype=torch.long, device=device)
        for row, sequence in enumerate(pending):
            input_ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            attention_mask[row, :len(sequence)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        if logits.ndim != 2 or logits.shape != (len(pending), 2):
            raise SafetyPipelineUnavailable(f"{_MODEL_NAMES[key]} returned an invalid score shape.")
        probabilities = torch.softmax(logits, dim=-1)[:, positive_index].cpu().tolist()
        for owner, raw in zip(owners, probabilities):
            score = _probability(raw, f"{_MODEL_NAMES[key]} score")
            scores[owner] = max(scores[owner], score)
        pending.clear()
        owners.clear()

    # Bounded forward batches avoid materializing a dataset-sized padded tensor.
    with _INFERENCE_LOCK, torch.inference_mode():
        for owner, text in enumerate(texts):
            windows = _token_windows(tokenizer, text)
            window_counts[owner] = len(windows)
            for window in windows:
                pending.append(window)
                owners.append(owner)
                if len(pending) == size:
                    infer_pending()
        if pending:
            infer_pending()
    return [_detector_result(key, score, count, device) for score, count in zip(scores, window_counts)]


def _score_detector(text: str, key: str) -> dict:
    return _score_detector_batch([text], key)[0]


def score_stage1_batch(texts: list[str], batch_size: int | None = None) -> list[dict[str, dict]]:
    """Score a group of prompts for validation/calibration, without a threshold.

Each output is the same model mapping returned by ``score_stage1_models``.
Use RUAS_STAGE1_DEVICE=cuda for offline calibration when the GPU is available;
the app defaults to CPU to leave memory available for chat and ModernBERT.
"""
    if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Prompts must be a list of non-empty strings.")
    if not texts:
        return []
    try:
        by_model = {key: _score_detector_batch(texts, key, batch_size) for key in STAGE1_MODEL_IDS}
        return [{key: results[index] for key, results in by_model.items()} for index in range(len(texts))]
    except SafetyPipelineUnavailable:
        raise
    except Exception as exc:
        raise SafetyPipelineUnavailable("Stage 1 requires valid scores from both prompt-injection detectors.") from exc


def score_stage1_models(text: str) -> dict[str, dict]:
    """Raw model results for runtime screening or validation-set calibration."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Prompt text must be a non-empty string.")
    try:
        return {key: _score_detector(text, key) for key in STAGE1_MODEL_IDS}
    except SafetyPipelineUnavailable:
        raise
    except Exception as exc:
        raise SafetyPipelineUnavailable("Stage 1 requires valid scores from both prompt-injection detectors.") from exc


def screen_prompt(text: str) -> dict:
    threshold, source = get_stage1_threshold()
    models = score_stage1_models(text)
    if set(models) != set(STAGE1_MODEL_IDS):
        raise SafetyPipelineUnavailable("Stage 1 requires both prompt-injection detectors.")
    mean = sum(_probability(result["unsafe_probability"], "detector score") for result in models.values()) / 2
    unsafe = mean >= threshold
    return {
        "label": "Unsafe" if unsafe else "Safe",
        "confidence": mean if unsafe else 1 - mean,
        "confidence_kind": "uncalibrated ensemble score",
        "unsafe_probability": mean,
        "threshold": threshold,
        "threshold_source": source,
        "models": models,
        "model": "Prompt Guard 2 + DeBERTa-v3 injection",
        "windows_scanned": max(result["windows_scanned"] for result in models.values()),
        "device": ", ".join(sorted({result["device"] for result in models.values()})),
        "ensemble": {"method": "average_probability", "aggregation": AGGREGATION},
    }


def _parse_llama_guard_output(content: str) -> tuple[str, list[str]]:
    if not isinstance(content, str):
        raise SafetyPipelineUnavailable("Llama Guard returned no text assessment.")
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if lines == ["safe"]:
        return "Safe", []
    if len(lines) != 2 or lines[0] != "unsafe" or not re.fullmatch(r"S\d+(?:\s*,\s*S\d+)*", lines[1]):
        raise SafetyPipelineUnavailable("Llama Guard returned an invalid safety/category assessment.")
    codes = list(dict.fromkeys(code.strip() for code in lines[1].split(",")))
    if any(code not in LLAMA_GUARD_CATEGORIES for code in codes):
        raise SafetyPipelineUnavailable("Llama Guard returned an unknown category code.")
    return "Unsafe", codes


def _guard_text_windows(text: str, max_bytes: int = 6000, overlap_chars: int = 128) -> list[str]:
    """Bound UTF-8 bytes, a conservative token ceiling, while retaining every character.

Llama 3 uses a byte-level BPE tokenizer. A 6,000-byte input leaves more than
2,000 tokens for Ollama's standard Guard 3 template and output in an 8K context.
"""
    chunks = []
    start = 0
    while start < len(text):
        end = start
        size = 0
        while end < len(text):
            next_size = len(text[end].encode("utf-8"))
            if size + next_size > max_bytes:
                break
            size += next_size
            end += 1
        if end == start:
            raise SafetyPipelineUnavailable("Llama Guard input window is too small.")
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def classify_llama_guard(text: str) -> dict:
    """Ask local Llama Guard for content categories without fabricating confidence."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Prompt text must be a non-empty string.")
    model = os.environ.get("RUAS_LLAMA_GUARD_MODEL", "llama-guard3:1b").strip()
    if not model:
        raise SafetyPipelineUnavailable("RUAS_LLAMA_GUARD_MODEL cannot be empty.")
    url = (os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434").strip().rstrip("/")
    windows = _guard_text_windows(text)
    category_codes = set()
    for window in windows:
        try:
            response = requests.post(
                f"{url}/api/chat",
                json={
                    "model": model,
                    # The installed Ollama Guard model supplies its moderation
                    # template. A generic assistant/system prompt would distort it.
                    "messages": [{"role": "user", "content": window}],
                    "stream": False,
                    "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "10m"),
                    "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 128},
                },
                timeout=(3, 120),
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or data.get("error") or data.get("done") is not True:
                raise SafetyPipelineUnavailable("Llama Guard did not complete its assessment.")
            if data.get("done_reason") == "length":
                raise SafetyPipelineUnavailable("Llama Guard output was truncated.")
            _, codes = _parse_llama_guard_output(data.get("message", {}).get("content"))
            category_codes.update(codes)
        except SafetyPipelineUnavailable:
            raise
        except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
            raise SafetyPipelineUnavailable(
                f"Llama Guard is unavailable. Start Ollama and install '{model}' with ollama pull."
            ) from exc
    categories = [{"code": code, "label": LLAMA_GUARD_CATEGORIES[code]} for code in LLAMA_GUARD_CATEGORIES if code in category_codes]
    return {
        "name": "Llama Guard",
        "model": model,
        "label": "Unsafe" if categories else "Safe",
        "categories": categories,
        "confidence": None,
        "confidence_kind": "category assessment; no probability provided",
        "predictions": [],
        "device": "ollama",
        "windows_scanned": len(windows),
    }
