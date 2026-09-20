"""Self-contained prompts, validation, and frozen Huatuo layer-20 extraction."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm


EXPECTED_REVISION = "6487c8071a3f87702a6816ba7a09f38921b685ed"
LOWER_LAYER_COUNT = 20
CODE_PROMPT = """Represent this individual clinical code as local evidence for medication decisions.
{evidence}
Local clinical evidence representation:"""
ROLE_TO_INDEX = {"diagnosis": 0, "procedure": 1}
CARD_ROLES = ("identity", "indications", "contraindications", "mechanism")
CARD_IDENTITY_PROMPT = """Represent this medication identity for clinical matching.
ATC class: {atc_code} — {title}
Generic ingredients: {ingredients}
Medication identity representation:"""
CARD_SECTION_PROMPT = """Represent this medication's sourced {role} knowledge for patient matching.
ATC class: {atc_code} — {title}
Complete {role} evidence:
{evidence}
Medication {role} representation:"""


def build_code_prompts(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    if [row["combined_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("clinical semantic rows are not in combined-index order")
    prompts, roles = [], []
    for row in rows:
        if row["kind"] not in ROLE_TO_INDEX:
            raise ValueError(f"unexpected code kind: {row['kind']}")
        prompt = CODE_PROMPT.format(evidence=row["prompt"])
        if row["prompt"] not in prompt:
            raise AssertionError("clinical code evidence was not preserved")
        prompts.append(prompt)
        roles.append(ROLE_TO_INDEX[row["kind"]])
    return prompts, np.asarray(roles, dtype=np.uint8)


def build_card_role_prompts(cards: list[dict]) -> list[list[str]]:
    result = [[] for _ in CARD_ROLES]
    for card in cards:
        common = {
            "atc_code": card["atc_code"],
            "title": card["official_atc_title"],
        }
        result[0].append(CARD_IDENTITY_PROMPT.format(
            **common,
            ingredients=", ".join(card.get("generic_ingredients", [])) or "unavailable",
        ))
        for role_index, role in enumerate(CARD_ROLES[1:], start=1):
            sections = []
            for subcard in card.get("subcards", []):
                content = subcard.get("sections", {}).get(role)
                if content:
                    ingredient = subcard.get("ingredient", "unspecified ingredient")
                    sections.append(f"Ingredient: {ingredient}\n{content}")
                    if content not in sections[-1]:
                        raise AssertionError("role content was not preserved")
            evidence = "\n\n".join(sections)
            if not evidence:
                evidence = f"No sourced {role} evidence is available."
            result[role_index].append(CARD_SECTION_PROMPT.format(
                **common, role=role, evidence=evidence,
            ))
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def load_cards(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = payload.get("signature")
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    if signature != hashlib.sha256(_canonical_json(unsigned)).hexdigest():
        raise ValueError("public medication-card signature mismatch")
    gate = payload.get("coverage_gate", {})
    if len(payload.get("cards", [])) != 125 or gate.get("passed") is not True:
        raise ValueError("public medication-card coverage gate is not qualified")
    target_flags = ("patient_outcomes_opened", "validation_opened", "test_opened")
    if any(payload.get(key) is not False for key in target_flags):
        raise ValueError("public cards violated target-independence")
    return payload


def chat_ids(tokenizer, prompts: list[str]) -> list[list[int]]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


@torch.inference_mode()
def encode_lower20(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    device: torch.device,
    *,
    description: str = "Huatuo encode",
) -> torch.Tensor:
    chunks = []
    starts = range(0, len(prompts), batch_size)
    progress = tqdm(
        starts,
        total=len(starts),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
    )
    for start in progress:
        ids = chat_ids(tokenizer, prompts[start : start + batch_size])
        encoded = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = model.model(
            **encoded, use_cache=False, return_dict=True
        ).last_hidden_state
        positions = encoded["attention_mask"].sum(1) - 1
        selected = hidden[
            torch.arange(len(positions), device=device), positions
        ]
        chunks.append(selected.cpu().to(torch.float16))
    return torch.cat(chunks)


def load_lower20(snapshot: Path, device: torch.device):
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Huatuo snapshot directory is missing: {snapshot}")
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to(device).eval().requires_grad_(False)
    if model.__class__.__name__ != "Qwen2ForCausalLM":
        raise ValueError("unexpected Huatuo model class")
    expected = {
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
    }
    for name, value in expected.items():
        if getattr(model.config, name) != value:
            raise ValueError(f"unexpected Huatuo {name}: {getattr(model.config, name)}")
    model.model.layers = torch.nn.ModuleList(
        list(model.model.layers[:LOWER_LAYER_COUNT])
    )
    model.config.num_hidden_layers = LOWER_LAYER_COUNT
    model.model.norm = torch.nn.Identity()
    model.lm_head = torch.nn.Identity()
    gc.collect()
    torch.cuda.empty_cache()
    if len(model.model.layers) != LOWER_LAYER_COUNT:
        raise AssertionError("lower-layer extraction boundary is wrong")
    return model
