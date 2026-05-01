"""Génère un fichier prompts.jsonl avec N prompts uniques répartis en 4 classes.

Classes (target distribution sur 1500 clips) :
  - indoor_manipulation 40% (600)  : main qui pose/saisit/verse un objet sur une surface
  - navigation_pov     25% (375)   : POV first-person, mouvement avant lent
  - outdoor_slow       20% (300)   : caméra fixe, scène calme avec mouvement lent
  - human_gesture      15% (225)   : close-up de main / geste expressif

Chaque prompt a aussi un `source_image_prompt` pour générer l'image source SDXL.
Output : prompts.jsonl avec un objet par ligne {id, class, prompt, source_image_prompt, seed}.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

# ---- Pools de slots --------------------------------------------------------

INDOOR_HAND_VERBS = [
    "picking up", "placing", "pouring from", "rotating", "sliding",
    "lifting", "tilting", "setting down",
]
INDOOR_OBJECTS = [
    "a glass of water", "a hardcover book", "a wooden cup",
    "a small bottle", "a ceramic mug", "a folded napkin",
    "a steel fork", "a ripe apple", "a yellow lemon",
    "a roll of tape", "a smartphone", "a pencil",
]
INDOOR_SURFACES = [
    "a wooden dining table", "a marble kitchen counter",
    "a polished oak desk", "a small round side table",
    "a beech cutting board", "a glass coffee table",
]
INDOOR_ROOMS = [
    "a sunlit kitchen", "a cozy living room", "a quiet home office",
    "a Scandinavian dining room", "a minimalist studio apartment",
]

NAV_LOCATIONS = [
    "a long hallway with neutral walls",
    "a bright corridor leading to a window",
    "a narrow kitchen passage between counters",
    "an empty open-plan living room",
    "a quiet office aisle with desks on both sides",
    "a clean stairwell with morning light",
]
NAV_MOTIONS = [
    "smooth slow forward dolly", "gentle handheld walking pace",
    "steady forward steady-cam", "subtle forward drift",
]

OUTDOOR_SCENES = [
    "a quiet park bench at golden hour",
    "an empty residential street at midday",
    "a cobbled square with a single passerby",
    "a small canal path with slow water",
    "a tree-lined boulevard with light traffic",
    "a community garden with gentle breeze",
]
OUTDOOR_SUBJECTS = [
    "one pedestrian walking by", "a slow cyclist passing",
    "leaves drifting in the wind", "a stray cat strolling",
    "a runner at moderate pace", "a child kicking a ball gently",
]

GESTURE_ACTIONS = [
    "a hand waving hello", "a finger pointing forward",
    "two hands clapping softly", "a hand opening an interior door",
    "a hand turning a light switch", "fingers typing on a laptop keyboard",
    "a hand holding a steaming mug", "a hand offering a small object",
]
GESTURE_BACKGROUNDS = [
    "a neutral beige background", "a softly blurred living room",
    "a clean white wall", "a warm wood-paneled wall",
]

STYLE_TAIL = (
    "natural lighting, photorealistic, shallow depth of field, "
    "subtle motion, 24 fps, cinematic"
)


def _build_indoor(rng: random.Random) -> tuple[str, str]:
    verb = rng.choice(INDOOR_HAND_VERBS)
    obj = rng.choice(INDOOR_OBJECTS)
    surface = rng.choice(INDOOR_SURFACES)
    room = rng.choice(INDOOR_ROOMS)
    prompt = (
        f"a hand {verb} {obj} on {surface} in {room}, {STYLE_TAIL}"
    )
    src = (
        f"close-up still photo of {obj} placed on {surface} in {room}, "
        f"natural daylight, photorealistic, sharp detail, no people"
    )
    return prompt, src


def _build_nav(rng: random.Random) -> tuple[str, str]:
    loc = rng.choice(NAV_LOCATIONS)
    motion = rng.choice(NAV_MOTIONS)
    prompt = (
        f"first person POV, {motion}, walking through {loc}, "
        f"natural ambient light, no other people in frame, {STYLE_TAIL}"
    )
    src = (
        f"first person view photo entering {loc}, "
        f"warm ambient lighting, photorealistic, no people"
    )
    return prompt, src


def _build_outdoor(rng: random.Random) -> tuple[str, str]:
    scene = rng.choice(OUTDOOR_SCENES)
    subj = rng.choice(OUTDOOR_SUBJECTS)
    prompt = (
        f"static camera tripod shot of {scene}, {subj}, "
        f"{STYLE_TAIL}"
    )
    src = (
        f"daylight photo of {scene}, photorealistic, calm composition, "
        f"single focal subject"
    )
    return prompt, src


def _build_gesture(rng: random.Random) -> tuple[str, str]:
    action = rng.choice(GESTURE_ACTIONS)
    bg = rng.choice(GESTURE_BACKGROUNDS)
    prompt = (
        f"close-up of {action}, on {bg}, "
        f"smooth natural motion, {STYLE_TAIL}"
    )
    src = (
        f"close-up still photo of a hand resting before performing the action, "
        f"on {bg}, photorealistic, soft natural light"
    )
    return prompt, src


CLASS_BUILDERS = {
    "indoor_manipulation": _build_indoor,
    "navigation_pov": _build_nav,
    "outdoor_slow": _build_outdoor,
    "human_gesture": _build_gesture,
}

DEFAULT_DISTRIBUTION = {
    "indoor_manipulation": 0.40,
    "navigation_pov": 0.25,
    "outdoor_slow": 0.20,
    "human_gesture": 0.15,
}


def generate(target: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[dict] = []
    next_id = itertools.count()
    counts = {k: int(round(v * target)) for k, v in DEFAULT_DISTRIBUTION.items()}
    # Ajuste pour totaliser exactement target
    diff = target - sum(counts.values())
    counts["indoor_manipulation"] += diff
    for cls, n in counts.items():
        builder = CLASS_BUILDERS[cls]
        attempts = 0
        added = 0
        while added < n and attempts < n * 30:
            prompt, src = builder(rng)
            h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            if h in seen:
                attempts += 1
                continue
            seen.add(h)
            seed_val = rng.randint(0, 2**31 - 1)
            out.append(
                {
                    "id": f"clip_{next(next_id):05d}",
                    "class": cls,
                    "prompt": prompt,
                    "source_image_prompt": src,
                    "seed": seed_val,
                }
            )
            added += 1
            attempts += 1
        if added < n:
            print(f"[warn] only {added}/{n} unique prompts for class {cls}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default="scripts/dataset_v3/prompts.jsonl",
    )
    args = p.parse_args()

    prompts = generate(args.target, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in prompts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} prompts to {out_path}")
    counts: dict[str, int] = {}
    for obj in prompts:
        counts[obj["class"]] = counts.get(obj["class"], 0) + 1
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n} ({n / len(prompts):.1%})")


if __name__ == "__main__":
    main()
