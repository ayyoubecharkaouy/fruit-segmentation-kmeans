#!/usr/bin/env python3
"""Télécharger une sélection reproductible de Fruits-360, sans cloner les dépôts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIELDS = [
    "image_id", "filename", "source", "fruit_types", "background", "lighting",
    "difficulty", "scene_id", "split", "source_file", "source_url", "commit",
    "license", "width", "height", "sha256", "annotation_status", "review_status",
    "original_filename", "dataset_title", "provenance_status",
]
LIMIT = 25 * 1024 * 1024


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_payload(response, url):
    length = response.headers.get("Content-Length")
    expected = int(length) if length is not None else None
    if expected is not None and expected > LIMIT:
        raise ValueError(f"Fichier trop volumineux : {url}")
    chunks, size = [], 0
    while True:
        chunk = response.read(min(64 * 1024, LIMIT + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > LIMIT:
            raise ValueError(f"Fichier trop volumineux : {url}")
    payload = b"".join(chunks)
    if expected is not None and size != expected:
        raise http.client.IncompleteRead(payload, expected - size)
    return payload


def fetch(url, *, parse_json=False):
    headers = {
        "User-Agent": "fruit-segmentation-mini-project/1.0",
        "Accept-Encoding": "identity",
    }
    if parse_json:
        headers["Accept"] = "application/vnd.github+json"
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = read_payload(response, url)
            # Valider dans la boucle pour réessayer aussi un JSON tronqué,
            # même lorsque Content-Length est absent ou incorrect côté serveur.
            return json.loads(payload) if parse_json else payload
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"HTTP {exc.code} pour {url}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException,
                json.JSONDecodeError, UnicodeDecodeError) as exc:
            if attempt == 2:
                problem = "Réponse JSON invalide ou incomplète" if isinstance(
                    exc, (json.JSONDecodeError, UnicodeDecodeError)
                ) else "Téléchargement interrompu ou réseau indisponible"
                raise RuntimeError(f"{problem} : {url} ({exc})") from exc
        print(f"Nouvelle tentative {attempt + 2}/3 : {url}", file=sys.stderr, flush=True)
        time.sleep(attempt + 1)
    raise RuntimeError(f"Téléchargement impossible : {url}")


def api(repository, endpoint):
    return fetch(f"https://api.github.com/repos/{repository}/{endpoint}", parse_json=True)


def directory_files(repository, root_tree, directory, cache):
    """Parcourir des arbres Git non récursifs, plus compacts que Contents."""
    def entries(tree_sha):
        key = repository, tree_sha
        if key not in cache:
            result = api(repository, "git/trees/" + tree_sha)
            if not isinstance(result, dict) or not isinstance(result.get("tree"), list):
                raise ValueError(f"Réponse d'arbre Git invalide : {repository}/{tree_sha}")
            if result.get("truncated"):
                raise ValueError(f"Liste de fichiers tronquée : {repository}/{tree_sha}")
            cache[key] = result["tree"]
        return cache[key]

    tree_sha = root_tree
    for part in directory.split("/"):
        child = next((entry for entry in entries(tree_sha)
                      if entry["path"] == part and entry["type"] == "tree"), None)
        if child is None:
            raise ValueError(f"Dossier introuvable : {repository}/{directory}")
        tree_sha = child["sha"]
    return [
        {"name": entry["path"], "path": f'{directory}/{entry["path"]}', "sha": entry["sha"]}
        for entry in entries(tree_sha)
        if entry["type"] == "blob" and entry.get("mode") != "120000"
    ]


def raw_url(repository, commit, path):
    return f"https://raw.githubusercontent.com/{repository}/{commit}/{urllib.parse.quote(path, safe='/')}"


def image_info(payload):
    from PIL import Image
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if min(image.size) < 20:
            raise ValueError("Image trop petite")
        if image.mode != "RGB":
            raise ValueError(f"Une image RGB était attendue, mode reçu : {image.mode}")
        return image.size


def validate_items(items):
    ids, filenames, partitions = set(), set(), defaultdict(set)
    for item in items:
        path = Path(item["filename"])
        if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("raw",):
            raise ValueError(f"Chemin invalide : {path}")
        if item["image_id"] in ids or item["filename"] in filenames:
            raise ValueError("Identifiant ou fichier dupliqué dans la sélection")
        if item["split"] not in ("mise_au_point", "evaluation"):
            raise ValueError("Partition inconnue")
        ids.add(item["image_id"])
        filenames.add(item["filename"])
        partitions[item["scene_id"]].add(item["split"])
    if any(len(value) != 1 for value in partitions.values()):
        raise ValueError("Un groupe de prises de vue est présent dans les deux partitions")


def build_manifest(plan, previous=None):
    previous = previous or {"commits": {}, "items": []}
    previous_items = {(item["source"], item["source_file"]): item for item in previous["items"]}
    next_id = max((int(re.fullmatch(r"img(\d+)", item["image_id"])[1])
                   for item in previous["items"]), default=0) + 1
    reserved_ids = {item["image_id"] for item in plan.get("local_items", [])}
    commits, root_trees = {}, {}
    for source_id, source in plan["sources"].items():
        commit = previous["commits"].get(source_id)
        if commit is None:
            reference = urllib.parse.quote(source["ref"], safe="/")
            commit = api(source["repository"], "git/ref/heads/" + reference)["object"]["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Révision GitHub invalide")
        commits[source_id] = commit
        # Le point d'accès /commits/{ref} inclut les fichiers et leurs diff.
        # /git/commits/{sha} ne renvoie que les métadonnées de l'objet Git.
        root_trees[source_id] = api(source["repository"], "git/commits/" + commit)["tree"]["sha"]

    items, folders, trees = [], {}, {}
    for group in plan["groups"]:
        source_id = group["source"]
        source = plan["sources"][source_id]
        repository, commit = source["repository"], commits[source_id]
        directory = group["directory"]
        key = source_id, directory
        if key not in folders:
            print(f"Sélection : {directory}", flush=True)
            folders[key] = directory_files(repository, root_trees[source_id], directory, trees)
        files = [entry for entry in folders[key] if entry["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
        if "filename_pattern" in group:
            pattern = re.compile(group["filename_pattern"], re.IGNORECASE)
            files = [entry for entry in files if pattern.search(entry["name"])]
        if "files" in group:
            by_name = {entry["name"]: entry for entry in files}
            missing = set(group["files"]) - by_name.keys()
            if missing:
                raise ValueError(f"Fichiers sources introuvables : {sorted(missing)}")
            selected = [by_name[name] for name in group["files"]]
        else:
            files.sort(key=lambda entry: hashlib.sha256(
                f'{plan["seed"]}:{group["id"]}:{entry["name"]}'.encode()
            ).hexdigest())
            selected = files[:group["count"]]
        if len(selected) != group["count"]:
            raise ValueError(f"Pas assez d'images pour le groupe {group['id']} dans {directory} : "
                             f"{len(selected)}/{group['count']}. Vérifier le filtre dans le plan.")
        for entry in selected:
            existing = previous_items.get((source_id, entry["path"]))
            if existing:
                image_id = existing["image_id"]
            else:
                while f"img{next_id:03d}" in reserved_ids:
                    next_id += 1
                image_id = f"img{next_id:03d}"
                next_id += 1
            suffix = Path(entry["name"]).suffix.lower()
            relative = existing["filename"] if existing else f'raw/{source_id}/{image_id}{suffix}'
            items.append({
                "image_id": image_id, "filename": relative, "source": source_id,
                "fruit_types": group["fruit_types"], "background": group["background"],
                "lighting": "a_verifier", "difficulty": group["difficulty"],
                "scene_id": group.get("scene_id", group["id"]), "split": group["split"],
                "source_file": entry["path"],
                "source_url": raw_url(repository, commit, entry["path"]),
                "commit": commit, "license": source["license"],
                "git_blob_sha": entry["sha"], "annotation_status": "non_annote",
                "review_status": "a_verifier_visuellement",
            })
            if existing:
                for field in ("background", "lighting", "difficulty", "review_status", "annotation_status"):
                    items[-1][field] = existing[field]
    validate_items(items)
    items.extend(dict(item) for item in plan.get("local_items", []))
    validate_items(items)
    return {"seed": plan["seed"], "commits": commits, "items": items}


def verify_payload(item, payload):
    checksum = hashlib.sha256(payload).hexdigest()
    if item.get("storage") == "local":
        if not item.get("sha256") or checksum != item["sha256"]:
            raise ValueError("Le fichier ajouté localement a changé depuis son inventaire")
    else:
        blob_sha = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if blob_sha != item["git_blob_sha"]:
            raise ValueError("Le fichier ne correspond pas à la version GitHub sélectionnée")
    width, height = image_info(payload)
    return {"width": width, "height": height, "sha256": checksum}


def audit(manifest):
    records, missing, invalid = [], [], []
    for item in manifest["items"]:
        path = DATA / item["filename"]
        if not path.is_file():
            missing.append(item["image_id"])
            continue
        try:
            info = verify_payload(item, path.read_bytes())
            records.append({**item, **info})
        except Exception as exc:
            invalid.append({"image_id": item["image_id"], "error": str(exc)})
    checksums = defaultdict(list)
    for record in records:
        checksums[record["sha256"]].append(record["image_id"])
    duplicates = [ids for ids in checksums.values() if len(ids) > 1]
    return records, missing, invalid, duplicates


def save_inventory(records):
    path = DATA / "metadata.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            exported = dict(record)
            exported.setdefault("original_filename", Path(record.get("source_file", "")).name)
            exported.setdefault("dataset_title", {
                "fruits360_simple": "Fruits-360 100x100",
                "fruits360_multi": "Fruits-360 multi",
            }.get(record["source"], record["source"]))
            exported.setdefault("provenance_status", "source_versionnee" if record.get("git_blob_sha") else "a_documenter")
            writer.writerow(exported)
    temporary.replace(path)


def contact_sheet(records):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    columns, cell_w, padding, gap, margin = 5, 244, 12, 16, 24
    preview_h, header_h, line_h = 170, 100, 20
    rows = (len(records) + columns - 1) // columns
    if not rows:
        return

    def font(size, bold=False):
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    title_font, id_font, body_font = font(24, True), font(16, True), font(14)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def wrap_caption(text):
        lines, current = [], ""
        for word in text.replace("|", ", ").split():
            candidate = f"{current} {word}".strip()
            if current and measure.textlength(candidate, font=body_font) > cell_w - 2 * padding:
                lines.append(current)
                current = word
            else:
                current = candidate
        return lines + [current]

    captions = [wrap_caption(record["fruit_types"]) for record in records]
    caption_lines = max(map(len, captions))
    cell_h = padding + preview_h + 12 + 24 + caption_lines * line_h + 10 + 24 + padding
    width = 2 * margin + columns * cell_w + (columns - 1) * gap
    height = header_h + rows * cell_h + (rows - 1) * gap + margin
    sheet = Image.new("RGB", (width, height), "#f3f4f6")
    draw = ImageDraw.Draw(sheet)
    counts = Counter(record["split"] for record in records)
    draw.text((margin, 20), "Jeu d'images - aperçu", font=title_font, fill="#111827")
    summary = (f"{len(records)} images | {counts['mise_au_point']} mise au point | "
               f"{counts['evaluation']} évaluation")
    draw.text((margin, 57), summary, font=body_font, fill="#4b5563")

    for index, record in enumerate(records):
        x = margin + index % columns * (cell_w + gap)
        y = header_h + index // columns * (cell_h + gap)
        draw.rounded_rectangle((x, y, x + cell_w - 1, y + cell_h - 1), radius=8,
                               fill="white", outline="#d1d5db")
        with Image.open(DATA / record["filename"]) as image:
            preview = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"),
                                       (cell_w - 2 * padding, preview_h))
        sheet.paste(preview, (x + (cell_w - preview.width) // 2,
                             y + padding + (preview_h - preview.height) // 2))
        text_y = y + padding + preview_h + 12
        draw.text((x + padding, text_y), record["image_id"], font=id_font, fill="#111827")
        for offset, caption in enumerate(captions[index]):
            draw.text((x + padding, text_y + 24 + offset * line_h), caption,
                      font=body_font, fill="#374151")
        evaluation = record["split"] == "evaluation"
        label = "Évaluation" if evaluation else "Mise au point"
        badge_y = y + cell_h - padding - 24
        badge_w = int(draw.textlength(label, font=body_font)) + 16
        draw.rounded_rectangle((x + padding, badge_y, x + padding + badge_w, badge_y + 24),
                               radius=5, fill="#dcfce7" if evaluation else "#dbeafe")
        draw.text((x + padding + 8, badge_y + 3), label, font=body_font,
                  fill="#166534" if evaluation else "#1e40af")
    sheet.save(DATA / "apercu.jpg", quality=95, subsampling=0)


def save_report(manifest, errors=()):
    records, missing, invalid, duplicates = audit(manifest)
    missing_licenses = [source_id for source_id in manifest["commits"]
                        if not (DATA / "licenses" / f"{source_id}.txt").is_file()]
    save_inventory(records)
    contact_sheet(records)
    managed = {item["filename"] for item in manifest["items"]}
    untracked = sorted(str(path.relative_to(DATA)) for path in (DATA / "raw").rglob("*")
                       if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                       and str(path.relative_to(DATA)) not in managed)
    provenance_pending = [item["image_id"] for item in manifest["items"]
                          if not item.get("source_url") or item.get("license") in (None, "", "a_verifier")]
    report = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "telecharge_a_revoir" if not (missing or invalid or duplicates or missing_licenses or untracked or errors) else "incomplet",
        "planned_images": len(manifest["items"]), "available_images": len(records),
        "by_split": dict(Counter(row["split"] for row in records)),
        "by_source": dict(Counter(row["source"] for row in records)),
        "missing_images": missing, "invalid_images": invalid,
        "missing_licenses": missing_licenses,
        "duplicate_groups": duplicates, "download_errors": list(errors),
        "untracked_images": untracked, "provenance_pending": provenance_pending,
        "visual_review_required": True, "ground_truth_masks_available": False,
    }
    write_json(DATA / "preparation_status.json", report)
    print(f'{len(records)}/{len(manifest["items"])} images disponibles et vérifiées.')
    if duplicates:
        print("Doublons exacts détectés ; voir preparation_status.json.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Afficher le plan sans accéder au réseau")
    parser.add_argument("--check", action="store_true", help="Vérifier les fichiers déjà téléchargés, sans réseau")
    args = parser.parse_args()
    plan = json.loads((DATA / "selection_plan.json").read_text(encoding="utf-8"))
    total = sum(group["count"] for group in plan["groups"]) + len(plan.get("local_items", []))
    if args.plan:
        by_split = Counter()
        for group in plan["groups"]:
            by_split[group["split"]] += group["count"]
        by_split.update(item["split"] for item in plan.get("local_items", []))
        print(json.dumps({"planned_images": total, "by_split": dict(by_split)}, ensure_ascii=False, indent=2))
        return 0
    metadata_path = DATA / "metadata.csv"
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        save_inventory([])
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("Installer Pillow : python -m pip install -r requirements-data.txt", file=sys.stderr)
        return 1
    manifest_path = DATA / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_items(manifest["items"])
    elif args.check:
        print("Aucune sélection téléchargée : le manifeste n'a pas encore été créé.", file=sys.stderr)
        return 1
    else:
        try:
            print("Résolution des versions et sélection des fichiers sources…", flush=True)
            manifest = build_manifest(plan)
            write_json(manifest_path, manifest)
        except Exception as exc:
            write_json(DATA / "preparation_status.json", {
                "updated_at": datetime.now(timezone.utc).isoformat(), "status": "selection_non_resolue",
                "planned_images": total, "available_images": 0, "error": str(exc),
                "visual_review_required": True, "ground_truth_masks_available": False,
            })
            print(str(exc), file=sys.stderr)
            return 1
    errors = []
    if not args.check:
        for source_id, source in plan["sources"].items():
            license_path = DATA / "licenses" / f"{source_id}.txt"
            if not license_path.exists():
                try:
                    payload = fetch(raw_url(source["repository"], manifest["commits"][source_id], "LICENSE"))
                    license_path.parent.mkdir(parents=True, exist_ok=True)
                    license_path.write_bytes(payload)
                except Exception as exc:
                    errors.append({"license": source_id, "error": str(exc)})
        for index, item in enumerate(manifest["items"], 1):
            path = DATA / item["filename"]
            try:
                if path.exists():
                    verify_payload(item, path.read_bytes())
                    print(f'[{index}/{len(manifest["items"])}] {item["image_id"]} déjà vérifiée', flush=True)
                    continue
                if item.get("storage") == "local":
                    raise ValueError("Ajout local absent : restaurer le fichier d'origine ; "
                                     "le lien du jeu Kaggle n'est pas une URL de téléchargement de cette image")
                payload = fetch(item["source_url"])
                verify_payload(item, payload)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".part")
                temporary.write_bytes(payload)
                temporary.replace(path)
                print(f'[{index}/{len(manifest["items"])}] {item["image_id"]} téléchargée', flush=True)
            except Exception as exc:
                errors.append({"image_id": item["image_id"], "error": str(exc)})
                print(f'{item["image_id"]} : {exc}', file=sys.stderr, flush=True)
            finally:
                # Conserver l'inventaire même si la session est interrompue ensuite.
                if index % 10 == 0:
                    records, _, _, _ = audit(manifest)
                    save_inventory(records)
    report = save_report(manifest, errors)
    return 0 if report["status"] == "telecharge_a_revoir" else 1


if __name__ == "__main__":
    raise SystemExit(main())
