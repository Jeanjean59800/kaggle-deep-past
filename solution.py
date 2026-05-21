# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
#!/usr/bin/env python3
"""
ByT5 Improved Inference with Host Guidance Preprocessing
=========================================================
Based on jeanjean111's high-scoring kernel (32.6), with additions:
- ASCII→diacritic normalization (per competition host guidance)
- Gap marker normalization (x→<gap>, x x x→<big_gap>)
- Optional consonantal skeleton preprocessing
- Repetition penalty to prevent loops

Competition: Deep Past Initiative Machine Translation
v1.0.0: Host guidance preprocessing on competitor's model
"""

import os
import re
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

os.environ["TQDM_DISABLE"] = "0"  # Keep progress bars for this one

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Use competitor's fine-tuned model
    "model_path": "/kaggle/input/000-byt5-base-big-new-align-tf32-cfg3",
    "test_path": "/kaggle/input/deep-past-initiative-machine-translation/test.csv",
    "batch_size": 8,
    "max_length": 512,
    "num_beams": 4,
    "early_stopping": True,
    # NEW: Our improvements
    "use_host_normalization": True,  # ASCII→diacritics, gap normalization
    "use_consonantal": False,  # Toggle to test consonantal impact
    "repetition_penalty": 1.21,  # Prevent repetition loops
    "min_length" : 15,
    "length_penalty" : 1.09,
}

print(f"Config: {CONFIG}")

# ============================================================
# HOST GUIDANCE: ASCII → DIACRITIC NORMALIZATION
# ============================================================

ASCII_TO_DIACRITIC = {
    # Sibilants
    "sz": "š",
    "SZ": "Š",
    "Sz": "Š",
    "sh": "š",
    "SH": "Š",
    "Sh": "Š",
    # Emphatics (comma notation)
    "s,": "ṣ",
    "S,": "Ṣ",
    "t,": "ṭ",
    "T,": "Ṭ",
    "z,": "ẓ",
    "Z,": "Ẓ",
    # Emphatics (dot-under notation)
    ".s": "ṣ",
    ".S": "Ṣ",
    ".t": "ṭ",
    ".T": "Ṭ",
    ".z": "ẓ",
    ".Z": "Ẓ",
    # H variants
    "h,": "ḫ",
    "H,": "Ḫ",
    ".h": "ḫ",
    ".H": "Ḫ",
    "hh": "ḫ",
    "HH": "Ḫ",
    # Aleph
    "'": "ʾ",
    # Subscript numbers (common ASCII alternatives)
    "s2": "š",
    "S2": "Š",
    "s3": "ś",
    "S3": "Ś",
}

VOWEL_SUBSCRIPTS = {
    "a2": "á",
    "a3": "à",
    "e2": "é",
    "e3": "è",
    "i2": "í",
    "i3": "ì",
    "u2": "ú",
    "u3": "ù",
    "A2": "Á",
    "A3": "À",
    "E2": "É",
    "E3": "È",
    "I2": "Í",
    "I3": "Ì",
    "U2": "Ú",
    "U3": "Ù",
}


def normalize_ascii_to_diacritics(text: str) -> str:
    """Convert ASCII transliteration conventions to proper diacritics."""
    result = text
    for ascii_seq, diacritic in sorted(
        ASCII_TO_DIACRITIC.items(), key=lambda x: -len(x[0])
    ):
        result = result.replace(ascii_seq, diacritic)
    for ascii_seq, diacritic in VOWEL_SUBSCRIPTS.items():
        result = result.replace(ascii_seq, diacritic)
    return result


def normalize_gaps(text: str) -> str:
    """Normalize gap/damage markers: x→<gap>, x x x→<big_gap>."""
    tokens = text.split()
    result = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if token.lower() == "x":
            x_count = 1
            while i + x_count < len(tokens) and tokens[i + x_count].lower() == "x":
                x_count += 1
            result.append("<gap>" if x_count == 1 else "<big_gap>")
            i += x_count
        else:
            if token.lower().startswith("x-"):
                result.append("<gap>" + token[1:])
            elif token.lower().endswith("-x"):
                result.append(token[:-1] + "-<gap>")
            else:
                result.append(token)
            i += 1

    text_result = " ".join(result)
    text_result = re.sub(r"<gap>\s+<big_gap>", "<big_gap>", text_result)
    text_result = re.sub(r"<big_gap>\s+<gap>", "<big_gap>", text_result)
    text_result = re.sub(r"(<gap>\s*){2,}", "<big_gap> ", text_result)
    return text_result.strip()


def apply_host_normalizations(text: str) -> str:
    """Apply all host-recommended normalizations."""
    text = normalize_ascii_to_diacritics(text)
    text = normalize_gaps(text)
    return text


# ============================================================
# CONSONANTAL SKELETON (optional)
# ============================================================

VOWEL_PATTERN = re.compile(r"[aeiuāēīūâêîûàèìùáéíúÀÈÌÙÁÉÍÚÂÊÎÛ]", re.IGNORECASE)
SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉ₓ"


def to_skeleton(text: str) -> str:
    """Convert to consonantal skeleton by removing vowels."""
    tokens = text.split()
    result = []
    for token in tokens:
        # Preserve logograms (uppercase), numbers, gap markers
        clean = "".join(c for c in token if c not in ".₀₁₂₃₄₅₆₇₈₉ₓ")
        if clean.isupper() or token.startswith("<"):
            result.append(token)
            continue
        try:
            float(token.replace(",", "."))
            result.append(token)
            continue
        except ValueError:
            pass
        # Strip vowels
        normalized = "".join(c for c in token if c not in SUBSCRIPTS)
        skeleton = VOWEL_PATTERN.sub("", normalized)
        skeleton = re.sub(r"-+", "", skeleton)
        result.append(skeleton if skeleton else "_")
    return " ".join(result)


def create_dual_input(text: str) -> str:
    """Create [SKL:skeleton] normalized format."""
    normalized = apply_host_normalizations(text)
    skeleton = to_skeleton(normalized)
    return f"[SKL:{skeleton}] {normalized}"


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

PREFIX = "translate Akkadian to English: "


def preprocess(text: str) -> str:
    """Apply all preprocessing based on config."""
    if CONFIG["use_host_normalization"]:
        text = apply_host_normalizations(text)

    if CONFIG["use_consonantal"]:
        skeleton = to_skeleton(text)
        text = f"[SKL:{skeleton}] {text}"

    return PREFIX + text


def postprocess_translation(text):
    # Return empty string for invalid outputs
    if not isinstance(text, str) or not text.strip():
        return ""

    # Normalize specific diacritics
    processed_text = text.replace("ḫ", "h").replace("Ḫ", "H")

    # Convert subscript digits to normal digits
    sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    processed_text = processed_text.translate(sub_map)

    # Normalize gap markers
    processed_text = re.sub(r"(\[x\]|\(x\)|\bx\b)", "<gap>", processed_text, flags=re.I)
    processed_text = re.sub(r"(\.{3,}|…|\[\.+\])", "<big_gap>", processed_text)

    # Merge adjacent gaps
    processed_text = re.sub(r"<gap>\s*<gap>", " <big_gap> ", processed_text)
    processed_text = re.sub(r"<big_gap>\s*<big_gap>", " <big_gap> ", processed_text)

    # Remove some parenthetical morphology notes
    processed_text = re.sub(
        r"\((fem|plur|pl|sing|singular|plural|\?|!)\.?\s*\w*\)",
        "",
        processed_text,
        flags=re.I,
    )

    # Temporarily protect tokens from cleanup
    processed_text = processed_text.replace("<gap>", "\x00GAP\x00")
    processed_text = processed_text.replace("<big_gap>", "\x00BIG\x00")

    # Remove bad characters
    bad_chars = '!?()"—–<>⌈⌋⌊[]+ʾ/;'
    processed_text = processed_text.translate(str.maketrans("", "", bad_chars))

    # Restore tokens
    processed_text = processed_text.replace("\x00GAP\x00", " <gap> ")
    processed_text = processed_text.replace("\x00BIG\x00", " <big_gap> ")

    # Handle common fractions from decimals
    frac_map = {
        r"\.5\b": " ½",
        r"\.25\b": " ¼",
        r"\.75\b": " ¾",
        r"\.33+\d*\b": " ⅓",
        r"\.66+\d*\b": " ⅔",
    }
    for pattern, replacement in frac_map.items():
        processed_text = re.sub(r"(\d+)" + pattern, r"\1" + replacement, processed_text)
        processed_text = re.sub(r"\b0" + pattern, replacement.strip(), processed_text)

    # Remove repeated single words
    processed_text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", processed_text)

    # Remove repeated n-grams
    for n in range(4, 1, -1):
        pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
        processed_text = re.sub(pattern, r"\1", processed_text)

    # Normalize whitespace and trim
    processed_text = re.sub(r"\s+", " ", processed_text).strip().strip("-")

    return processed_text

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nLoading model from: {CONFIG['model_path']}")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_path"]).to(DEVICE)
model.eval()
model.float()  # FP32 as in original

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

test_df = pd.read_csv(CONFIG["test_path"])
print(f"Test samples: {len(test_df)}")

# ============================================================
# SHOW PREPROCESSING EXAMPLES
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING EXAMPLES")
print("=" * 60)

for i in range(min(3, len(test_df))):
    original = test_df["transliteration"].iloc[i]
    processed = preprocess(original)
    print(f"\n[{i + 1}] Original:  {original[:70]}...")
    print(f"    Processed: {processed[:70]}...")

# ============================================================
# INFERENCE DATASET
# ============================================================


class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer, preprocess_fn):
        self.texts = [
            preprocess_fn(t) for t in df["transliteration"].astype(str).tolist()
        ]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            max_length=CONFIG["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


test_dataset = InferenceDataset(test_df, tokenizer, preprocess)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

# ============================================================
# GENERATE TRANSLATIONS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING TRANSLATIONS")
print("=" * 60)

all_predictions = []

torch.set_grad_enabled(False)

with torch.inference_mode():
    for batch in tqdm(test_loader, desc="Translating"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CONFIG["max_length"],
            num_beams=CONFIG["num_beams"],
            early_stopping=CONFIG["early_stopping"],
            repetition_penalty=CONFIG["repetition_penalty"],  # NEW: prevent loops
            min_length=CONFIG["min_length"], 
            length_penalty = CONFIG["length_penalty"]
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend([postprocess_translation(d) for d in decoded])

# ============================================================
# SAMPLE OUTPUTS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE TRANSLATIONS")
print("=" * 60)

for i in range(min(5, len(all_predictions))):
    src = test_df["transliteration"].iloc[i][:60]
    tgt = all_predictions[i][:100]
    print(f"\n[{i + 1}] Source: {src}...")
    print(f"    Translation: {tgt}")


# ============================================================
# CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "translation": all_predictions,
    }
)

# Handle empty predictions
submission["translation"] = submission["translation"].apply(
    lambda x: x if len(x) > 0 else "broken text"
)

submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved: {len(submission)} rows")
print(submission.head())

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("INFERENCE COMPLETE")
print(f"Model: {CONFIG['model_path']}")
print(f"Host normalization: {CONFIG['use_host_normalization']}")
print(f"Consonantal: {CONFIG['use_consonantal']}")
print(f"Repetition penalty: {CONFIG['repetition_penalty']}")
print("=" * 60)
