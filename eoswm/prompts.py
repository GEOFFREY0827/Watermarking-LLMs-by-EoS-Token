"""Prompt sources. The built-in pool works offline; any HF dataset or a local file can replace it."""
from __future__ import annotations

import json
import os

TOPICS = [
    "the moon", "volcanoes", "honeybees", "the printing press", "black holes", "rainforests",
    "the Roman Empire", "photosynthesis", "the stock market", "glaciers", "octopuses", "jazz",
    "electric cars", "the human heart", "coffee", "earthquakes", "the internet", "penguins",
    "solar panels", "the Great Wall of China", "vaccines", "deserts", "chess", "the ocean tides",
    "dinosaurs", "bread baking", "the piano", "hurricanes", "satellites", "bamboo", "rainbows",
    "the Olympic Games", "recycling", "the brain", "tea ceremonies", "lighthouses", "coral reefs",
    "the bicycle", "wind turbines", "mushrooms", "the Silk Road", "thunderstorms", "elephants",
    "origami", "the periodic table", "submarines", "wolves", "the Renaissance", "batteries",
    "the Sahara", "maple syrup", "whales", "telescopes", "the Amazon river", "bridges", "owls",
    "clocks", "the immune system", "kites", "the Arctic", "paper money", "trains", "sharks",
    "soap", "the violin", "comets", "cotton", "pyramids", "frogs", "the compass", "snowflakes",
    "democracy", "rice farming", "the camera", "butterflies", "the Pacific Ocean", "glass",
    "robots", "the alphabet", "cheese", "the solar system", "ants", "libraries", "salt",
    "airplanes", "the rainy season", "sunflowers", "the steam engine", "caves", "dolphins",
]

TEMPLATES = [
    "Describe {t} in two sentences.",
    "What is one interesting fact about {t}?",
    "Explain {t} to a ten-year-old in a short paragraph.",
    "Give a one-sentence summary of {t}.",
    "Why is {t} important? Answer briefly.",
    "Write a short haiku about {t}.",
    "List three words associated with {t}.",
    "What is a common misconception about {t}? Keep it short.",
    "In one sentence, how would you introduce {t} to a stranger?",
    "Name one question scientists still ask about {t}.",
    "Write a brief tweet about {t}.",
    "What does {t} have to do with everyday life? Answer in two sentences.",
]


def builtin_prompts() -> list[str]:
    return [tpl.format(t=t) for t in TOPICS for tpl in TEMPLATES]


def load_prompts(source: str = "builtin", limit: int | None = None) -> list[str]:
    """`builtin`, a .txt/.jsonl file (one prompt per line / field `prompt` or `instruction`),
    or `hf:<dataset>[:<split>]` (needs the `datasets` package and network)."""
    if source == "builtin":
        prompts = builtin_prompts()
    elif source.startswith("hf:"):
        from datasets import load_dataset  # optional dependency

        spec = source[3:].split(":")
        ds = load_dataset(spec[0], split=spec[1] if len(spec) > 1 else "train")
        field = next(f for f in ("prompt", "instruction", "question", "text") if f in ds.column_names)
        prompts = [str(x) for x in ds[field] if str(x).strip()]
    elif os.path.exists(source):
        with open(source, encoding="utf-8") as f:
            if source.endswith(".jsonl"):
                rows = [json.loads(line) for line in f if line.strip()]
                prompts = [r.get("prompt") or r.get("instruction") for r in rows]
            else:
                prompts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unknown prompt source: {source}")
    return prompts[:limit] if limit else prompts
