#!/usr/bin/env python3
"""Enregistrer et ranger les ajouts locaux, avec sauvegarde des anciens noms."""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

if __package__:
    from . import prepare_data as prep
else:
    import prepare_data as prep


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def organize():
    additions = load(prep.DATA / "manual_additions.json")["items"]
    manifest = load(prep.DATA / "download_manifest.json")
    prep.validate_items(manifest["items"])
    records, missing, invalid, duplicates = prep.audit(manifest)
    if missing or invalid or duplicates:
        raise ValueError("Vérifier les images déjà inventoriées avant l'import")
    by_id = {item["image_id"]: dict(item) for item in manifest["items"]}
    moves, changes = [], []
    for addition in additions:
        item = {key: value for key, value in addition.items() if key != "input_filename"}
        original = prep.DATA / addition["input_filename"]
        destination = prep.DATA / item["filename"]
        prep.validate_items([item])
        input_path = Path(addition["input_filename"])
        if input_path.is_absolute() or ".." in input_path.parts or input_path.parts[:1] != ("raw",):
            raise ValueError("Chemin d'import invalide")
        existing = by_id.get(item["image_id"])
        if existing and (existing.get("sha256") != item["sha256"] or existing["filename"] != item["filename"]):
            raise ValueError(f"Identifiant déjà utilisé : {item['image_id']}")
        if original.exists() and destination.exists() and original != destination:
            raise ValueError(f"L'ancien et le nouveau nom existent simultanément : {original}")
        source = original if original.exists() else destination
        if not source.is_file():
            raise ValueError(f"Ajout introuvable : {source}")
        payload = source.read_bytes()
        prep.verify_payload(item, payload)
        with Image.open(source) as image:
            extensions = {"JPEG": {".jpg", ".jpeg"}, "MPO": {".jpg", ".jpeg", ".mpo"}, "PNG": {".png"}}
            if destination.suffix.lower() not in extensions.get(image.format, set()):
                raise ValueError(f"Extension incohérente avec le contenu : {destination}")
        if source != destination:
            moves.append((source, destination))
            changes.append({"old_filename": str(source.relative_to(prep.DATA)),
                            "new_filename": item["filename"], "image_id": item["image_id"]})
        by_id[item["image_id"]] = item
    candidate = {**manifest, "items": sorted(by_id.values(), key=lambda item: int(item["image_id"][3:]))}
    prep.validate_items(candidate["items"])
    expected_names = {item["filename"] for item in candidate["items"]}
    original_names = {addition["input_filename"] for addition in additions}
    unexpected = [str(path.relative_to(prep.DATA)) for path in (prep.DATA / "raw").rglob("*")
                  if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                  and str(path.relative_to(prep.DATA)) not in expected_names | original_names]
    if unexpected:
        raise ValueError(f"Autres images à organiser explicitement : {unexpected}")
    if candidate == manifest and not moves:
        report = prep.save_report(candidate)
        if report["status"] != "telecharge_a_revoir":
            raise ValueError("La vérification du jeu organisé a échoué")
        print("Les ajouts sont déjà organisés.")
        return
    archive = prep.DATA / "history" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_organisation")
    archive.mkdir(parents=True)
    state_files = ["download_manifest.json", "selection_plan.json", "selection_plan_balanced.json",
                   "metadata.csv", "preparation_status.json", "apercu.jpg"]
    for name in state_files:
        if (prep.DATA / name).exists():
            shutil.copy2(prep.DATA / name, archive / name)
    prep.write_json(archive / "renaming_map.json", changes)
    moved = []
    try:
        for original, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            original.replace(destination)
            moved.append((original, destination))
        prep.write_json(prep.DATA / "download_manifest.json", candidate)
        for name in ("selection_plan.json", "selection_plan_balanced.json"):
            plan_path = prep.DATA / name
            if plan_path.exists():
                plan = load(plan_path)
                plan["local_items"] = [by_id[addition["image_id"]] for addition in additions]
                prep.write_json(plan_path, plan)
        report = prep.save_report(candidate)
        if report["status"] != "telecharge_a_revoir":
            raise ValueError("La vérification finale a échoué")
    except Exception:
        for original, destination in reversed(moved):
            original.parent.mkdir(parents=True, exist_ok=True)
            destination.replace(original)
        for name in state_files:
            if (archive / name).exists():
                shutil.copy2(archive / name, prep.DATA / name)
        raise
    prep.write_json(prep.DATA / "organisation_status.json", {
        "updated_at": datetime.now(timezone.utc).isoformat(), "status": "organise",
        "available_images": report["available_images"],
        "renamed_files": changes, "archive": str(archive.relative_to(prep.DATA)),
        "pending_licenses": report["provenance_pending"],
    })
    print(f"Organisation terminée : {len(changes)} fichiers rangés, {len(candidate['items'])} images inventoriées.")


if __name__ == "__main__":
    try:
        organize()
    except Exception as exc:
        print(f"Organisation interrompue : {exc}", file=sys.stderr)
        raise SystemExit(1)
