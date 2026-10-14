"""Generate and verify the synthetic stripe dataset; no model training.

Usage: python scripts/generate_stripes.py [--output data/stripes-v1]
       python scripts/generate_stripes.py --verify-only
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys

import numpy as np
import PIL
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSION = "1.0.0"
TRANSFORMS = ("identity", "transpose", "brightness", "invert", "scale")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_pixels(image):
    return hashlib.sha256(np.ascontiguousarray(image, dtype="<f4").tobytes()).hexdigest()


def rng(seed):
    return np.random.Generator(np.random.PCG64(seed))


def cells(config, split):
    heldout = tuple(config["heldout"])
    return [(w, eta) for w, eta in itertools.product(config["widths"], config["noise_amplitudes"])
            if ((w, eta) == heldout) == split["heldout_only"]]


def validate_config(config):
    require(config["version"] == "stripes-v1" and config["size"] == 32, "Expected stripes-v1, size 32")
    require(config["widths"] == [2, 4, 8], "Unexpected stripe widths")
    require(config["noise_amplitudes"] == [0.0, 0.03, 0.08], "Unexpected noise amplitudes")
    require(config["heldout"] == [8, 0.08], "Unexpected held-out combination")
    require(config["brightness_range"] == [0.1, 0.3] and config["contrast_range"] == [0.3, 0.5],
            "Unexpected brightness/contrast ranges")
    require(list(config["splits"]) == ["train", "validation", "test_id", "test_ood"], "Unexpected split order")
    seeds = []
    for name, split in config["splits"].items():
        require(type(split["per_cell_class"]) is int and split["per_cell_class"] > 0, "Invalid quota")
        require(type(split["seed"]) is int and split["seed"] >= 0, "Invalid seed")
        require(split["heldout_only"] is (name == "test_ood"), "Held-out combination must be isolated")
        seeds.append(split["seed"])
    require(len(set(seeds)) == len(seeds), "Split seeds must differ")
    require(config["diagnostics"] == {
        "brightness": [0.1, 0.3], "contrast": [0.3, 0.5],
        "brightness_offset": 0.05, "scale": 0.5, "constants": [0.0, 0.5, 1.0]},
        "Unexpected diagnostic definition")


def render_sample(row, size):
    # b and a are stored as exact float32 values represented in JSON.
    # Compute the expression and noise in float64, then round pixels once.
    t = np.arange(size, dtype=np.int64)
    q = ((t + row["phase"]) // row["width"]) % 2
    band = row["brightness"] + row["contrast"] * q
    clean = np.broadcast_to(band[:, None] if row["label"] == 0 else band[None, :], (size, size))
    noise = rng(row["noise_seed"]).uniform(-row["noise_amplitude"], row["noise_amplitude"], (size, size))
    return (clean + noise).astype("<f4")[None, :, :]


def family_key(row):
    return (row["width"], row["phase"], row["brightness"], row["contrast"])


def transform(image, name, config):
    spec = config["diagnostics"]
    if name == "identity":
        return image.copy()
    if name == "transpose":
        return image.transpose(0, 2, 1).copy()
    if name == "brightness":
        return (image.astype(np.float64) + spec["brightness_offset"]).astype("<f4")
    if name == "invert":
        return (1.0 - image.astype(np.float64)).astype("<f4")
    if name == "scale":
        return (image.astype(np.float64) * spec["scale"]).astype("<f4")
    raise ValueError(f"Unknown transform: {name}")


def diagnostic_bases(config):
    for width in config["widths"]:
        for phase, b, a, label in itertools.product(
            range(2 * width), config["diagnostics"]["brightness"], config["diagnostics"]["contrast"], (0, 1)
        ):
            yield {"width": width, "phase": phase, "brightness": float(np.float32(b)),
                   "contrast": float(np.float32(a)), "label": label,
                   "noise_amplitude": 0.0, "noise_seed": 0}


def make_diagnostics(config):
    images, rows, relations, index = [], [], [], {}

    def add(image, metadata):
        digest = sha_pixels(image)
        if digest in index:
            return index[digest]
        sample_id = f"diagnostics-{len(rows):05d}"
        index[digest] = sample_id
        images.append(image)
        rows.append({"id": sample_id, "split": "diagnostics", **metadata, "pixel_sha256": digest})
        return sample_id

    bases = []
    for params in diagnostic_bases(config):
        image = render_sample(params, config["size"])
        sample_id = add(image, {**params, "operation": "identity"})
        bases.append((sample_id, image, params))
    for base_id, image, params in bases:
        for operation in TRANSFORMS:
            label = 1 - params["label"] if operation == "transpose" else params["label"]
            target_id = add(transform(image, operation, config),
                            {**params, "label": label, "source_label": params["label"],
                             "operation": operation, "base_id": base_id})
            relations.append({"base_id": base_id, "target_id": target_id, "transform": operation})
    for value in config["diagnostics"]["constants"]:
        add(np.full((1, config["size"], config["size"]), value, dtype="<f4"),
            {"label": -1, "operation": "constant", "value": value})
    return np.stack(images), rows, relations


def save_bundle(output, name, images, rows):
    np.savez_compressed(output / f"{name}.npz", images=images,
                        labels=np.array([row["label"] for row in rows], dtype="<i8"),
                        sample_ids=np.array([row["id"] for row in rows]))
    write_rows(output / f"{name}.jsonl", rows)
    # Show two examples of each class/cell; diagnostics show each transformation.
    groups = {}
    for i, row in enumerate(rows):
        key = (row.get("operation", "sample"), row.get("width"), row.get("noise_amplitude"), row["label"])
        group = groups.setdefault(key, [])
        if len(group) < 2:
            group.append(i)
    selected = [i for key in sorted(groups, key=str) for i in groups[key]][:64]
    columns, tile_w, tile_h, header = 8, 140, 160, 38
    canvas = Image.new("RGB", (columns * tile_w, ((len(selected) + columns - 1) // columns) * tile_h + header), "#eef0f4")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"{name}: float32 data / PNG preview only; H=0, V=1, constant=-1", fill="#111827")
    for slot, i in enumerate(selected):
        x, y = (slot % columns) * tile_w, (slot // columns) * tile_h + header
        pixels = np.rint(images[i, 0] * 255).clip(0, 255).astype(np.uint8)
        tile = Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST)
        canvas.paste(tile, (x + 6, y))
        row = rows[i]
        label = {0: "H", 1: "V", -1: "constant"}[row["label"]]
        description = (f"{label} w={row['width']} n={row['noise_amplitude']:g}" if name != "diagnostics"
                       else f"{label} {row['operation']}")
        draw.text((x + 6, y + 131), description, fill="#111827")
        draw.text((x + 6, y + 143), row["id"], fill="#374151")
    canvas.save(output / f"{name}-preview.png")


def generate(config, output):
    validate_config(config)
    # Never silently replace a previous dataset, including a partial generation.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    diagnostic_images, diagnostic_rows, relations = make_diagnostics(config)
    save_bundle(output, "diagnostics", diagnostic_images, diagnostic_rows)
    write_rows(output / "diagnostic-relations.jsonl", relations)
    seen_pixels = {row["pixel_sha256"] for row in diagnostic_rows}
    seen_families = {family_key(row) for row in diagnostic_bases(config)}
    rejected = {}
    for name, split in config["splits"].items():
        random = rng(split["seed"])
        rows, images = [], []
        rejected[name] = 0
        for width, eta in cells(config, split):
            for label in (0, 1):
                for _ in range(split["per_cell_class"]):
                    for attempt in range(1000):
                        row = {"id": f"{name}-{len(rows):05d}", "split": name, "label": label,
                               "width": width, "phase": int(random.integers(0, 2 * width)),
                               "brightness": float(np.float32(random.uniform(*config["brightness_range"]))),
                               "contrast": float(np.float32(random.uniform(*config["contrast_range"]))),
                               "noise_amplitude": eta, "noise_seed": int(random.integers(0, 2**63))}
                        image = render_sample(row, config["size"])
                        key, digest = family_key(row), sha_pixels(image)
                        if key not in seen_families and digest not in seen_pixels:
                            break
                        rejected[name] += 1
                    else:
                        raise ValueError(f"Could not sample a unique image for {name}")
                    seen_families.add(key)
                    seen_pixels.add(digest)
                    rows.append({**row, "pixel_sha256": digest})
                    images.append(image)
        permutation = random.permutation(len(rows))
        save_bundle(output, name, np.stack(images)[permutation], [rows[i] for i in permutation])
    write_json(output / "manifest.json", {
        "generator_version": GENERATOR_VERSION, "generator_sha256": sha_file(Path(__file__)),
        "python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__,
        "rng": "numpy.random.Generator(PCG64)",
        "numeric_policy": "float32 b/a -> float64 formula and uniform noise -> little-endian float32 pixels",
        "shape": "N,C,H,W", "labels": {"0": "horizontal", "1": "vertical", "-1": "diagnostic constant only"},
        "diagnostic_base_count": 224, "diagnostic_relation_count": len(relations),
        "diagnostic_unique_count": len(diagnostic_rows), "resampled_collisions": rejected,
    })
    report = verify(output, check_hashes=False)
    write_json(output / "validation-report.json", report)
    write_json(output / "checksums.json", {path.name: sha_file(path) for path in sorted(output.iterdir())})
    return report


def verify(output, check_hashes=True):
    if check_hashes:
        checksums = read_json(output / "checksums.json")
        actual_files = {path.name for path in output.iterdir() if path.is_file()} - {"checksums.json"}
        require(set(checksums) == actual_files, "Checksum file list mismatch")
        for filename, digest in checksums.items():
            require(sha_file(output / filename) == digest, f"File checksum mismatch: {filename}")
    config = read_json(output / "config.json")
    validate_config(config)
    summary, seen_pixels, seen_families = {}, {}, {}
    bundles = {}
    for name in ["diagnostics", *config["splits"]]:
        rows = read_rows(output / f"{name}.jsonl")
        with np.load(output / f"{name}.npz", allow_pickle=False) as archive:
            require(set(archive.files) == {"images", "labels", "sample_ids"}, f"Invalid NPZ keys: {name}")
            images, labels, ids = archive["images"], archive["labels"], archive["sample_ids"]
        require(images.dtype == np.dtype("<f4") and images.shape == (len(rows), 1, 32, 32), f"Invalid images: {name}")
        require(labels.dtype == np.dtype("<i8") and labels.shape == (len(rows),), f"Invalid labels: {name}")
        require(ids.shape == (len(rows),) and len(set(ids.tolist())) == len(rows), f"Invalid IDs: {name}")
        require(np.isfinite(images).all() and images.min() >= 0 and images.max() <= 1, f"Invalid pixels: {name}")
        counts = Counter()
        for i, row in enumerate(rows):
            require(row["split"] == name and row["id"] == ids[i] and row["label"] == labels[i], f"Metadata mismatch: {name}/{i}")
            digest = sha_pixels(images[i])
            require(digest == row["pixel_sha256"], f"Pixel hash mismatch: {name}/{i}")
            require(digest not in seen_pixels, f"Repeated image: {name}/{i}")
            seen_pixels[digest] = name
            if name != "diagnostics":
                require(row["label"] in (0, 1), "Invalid class")
                require(row["width"] in config["widths"] and 0 <= row["phase"] < 2 * row["width"], "Invalid geometry")
                for field, limits in [("brightness", "brightness_range"), ("contrast", "contrast_range")]:
                    low, high = np.array(config[limits], dtype=np.float32)
                    require(low <= row[field] <= high, f"Invalid {field}")
                require(0 <= row["noise_seed"] < 2**63, "Invalid noise seed")
                key = family_key(row)
                require(key not in seen_families, f"Repeated sample family: {name}/{i}")
                seen_families[key] = name
                require(np.array_equal(images[i], render_sample(row, 32)), f"Sample replay mismatch: {name}/{i}")
                # Independently check the label's geometric axis and bounded residual.
                rr, cc = np.indices((32, 32))
                coordinate = rr if row["label"] == 0 else cc
                bright = np.floor_divide(coordinate + row["phase"], row["width"]) % 2 == 1
                clean = np.where(bright, row["brightness"] + row["contrast"], row["brightness"])
                require(np.abs(images[i, 0].astype(np.float64) - clean).max() <= row["noise_amplitude"] + 1e-7,
                        f"Geometry/noise bound mismatch: {name}/{i}")
                counts[(row["width"], row["noise_amplitude"], row["label"])] += 1
        if name != "diagnostics":
            spec = config["splits"][name]
            expected = Counter({(w, eta, label): spec["per_cell_class"] for w, eta in cells(config, spec) for label in (0, 1)})
            require(counts == expected, f"Class/cell quotas or held-out isolation failed: {name}")
            require(images.min() >= 0.02 - 1e-7 and images.max() <= 0.88 + 1e-7, "Classification pixel bounds failed")
        else:
            for row in diagnostic_bases(config):
                seen_families[family_key(row)] = "diagnostics"
        with Image.open(output / f"{name}-preview.png") as preview:
            preview.verify()
        summary[name] = {"count": len(rows), "class_counts": {str(k): v for k, v in sorted(Counter(labels.tolist()).items())},
                         "pixel_min": float(images.min()), "pixel_max": float(images.max()),
                         "cells": [{"width": w, "noise_amplitude": eta, "label": label, "count": count}
                                   for (w, eta, label), count in sorted(counts.items())]}
        bundles[name] = (images, rows)

    images, rows = bundles["diagnostics"]
    by_id = {row["id"]: (images[i], row) for i, row in enumerate(rows)}
    base_rows = [row for row in rows if row["operation"] == "identity"]
    expected_bases = {(family_key(row), row["label"]) for row in diagnostic_bases(config)}
    require({(family_key(row), row["label"]) for row in base_rows} == expected_bases and len(base_rows) == 224,
            "Diagnostic base coverage mismatch")
    for row in base_rows:
        require(np.array_equal(by_id[row["id"]][0], render_sample(row, 32)), "Diagnostic base formula mismatch")
    relations = read_rows(output / "diagnostic-relations.jsonl")
    expected_relations = {(row["id"], operation) for row in base_rows for operation in TRANSFORMS}
    require(len(relations) == len(expected_relations) and
            {(row["base_id"], row["transform"]) for row in relations} == expected_relations,
            "Diagnostic relation coverage mismatch")
    used = set()
    for relation in relations:
        source, metadata = by_id[relation["base_id"]]
        target, target_metadata = by_id[relation["target_id"]]
        require(np.array_equal(target, transform(source, relation["transform"], config)), "Diagnostic transform mismatch")
        expected_label = 1 - metadata["label"] if relation["transform"] == "transpose" else metadata["label"]
        require(target_metadata["label"] == expected_label, "Diagnostic transform label mismatch")
        used.add(relation["target_id"])
    constants = [row for row in rows if row["operation"] == "constant"]
    require(sorted(row["value"] for row in constants) == config["diagnostics"]["constants"], "Constant probes mismatch")
    for row in constants:
        require(row["label"] == -1 and np.all(by_id[row["id"]][0] == row["value"]), "Invalid constant probe")
        used.add(row["id"])
    require(used == set(by_id), "Unreferenced diagnostic image")
    return {"status": "passed", "splits": summary, "classification_count": sum(summary[n]["count"] for n in config["splits"]),
            "diagnostic_base_count": len(base_rows), "diagnostic_relation_count": len(relations),
            "checks": ["NPZ schema, dtype, shape and finite pixels", "metadata, pixel SHA256 and full noise replay",
                       "label geometry and bounded noise", "class/cell quotas and held-out isolation",
                       "no duplicate pixels across all bundles", "no repeated base families across classification splits or diagnostic bases",
                       "224 diagnostic bases and all transform relations", "constant labels excluded from classification", "PNG decoding"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stripes-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/stripes-v1")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.verify_only:
        report = verify(output)
    else:
        report = generate(read_json(args.config), output)
        require(verify(output) == report, "Post-write verification mismatch")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "classification_count": report["classification_count"],
                      "diagnostic_count": report["splits"]["diagnostics"]["count"]}, indent=2))


if __name__ == "__main__":
    main()
