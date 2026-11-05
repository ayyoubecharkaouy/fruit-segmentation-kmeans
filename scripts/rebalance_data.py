#!/usr/bin/env python3
"""Appliquer le plan équilibré sans remplacer le jeu actif avant vérification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

if __package__:
    from . import prepare_data as prep
else:
    import prepare_data as prep


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def statistics(plan):
    by_split, by_content = Counter(), Counter()
    for group in plan["groups"]:
        by_split[group["split"]] += group["count"]
        if group["source"] == "fruits360_multi":
            by_content[group["fruit_types"]] += group["count"]
    return {"target_images": sum(by_split.values()), "by_split": dict(by_split),
            "multi_by_content": dict(by_content)}


def candidate_manifest(plan, active):
    path = prep.DATA / "rebalance_manifest.json"
    plan_signature, base_signature = signature(plan), signature(active)
    if path.exists():
        candidate = load(path)
        if candidate.get("plan_sha256") == plan_signature and candidate.get("base_sha256") == base_signature:
            prep.validate_items(candidate["items"])
            return candidate
    candidate = prep.build_manifest(plan, previous=active)
    candidate.update(plan_sha256=plan_signature, base_sha256=base_signature)
    prep.write_json(path, candidate)
    return candidate


def stage_downloads(candidate):
    """Conserver les nouveaux octets hors de raw/ tant que la sélection est incomplète."""
    stage = prep.DATA / "staging" / "rebalance"
    checksums, staged = defaultdict(list), []
    for index, item in enumerate(candidate["items"], 1):
        local = prep.DATA / item["filename"]
        pending = stage / item["filename"]
        if local.exists():
            payload = local.read_bytes()
            state = "conservée"
        elif pending.exists():
            payload = pending.read_bytes()
            state = "déjà récupérée"
        else:
            payload = prep.fetch(item["source_url"])
            prep.verify_payload(item, payload)
            pending.parent.mkdir(parents=True, exist_ok=True)
            temporary = pending.with_suffix(pending.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(pending)
            state = "récupérée"
        info = prep.verify_payload(item, payload)
        checksums[info["sha256"]].append(item["image_id"])
        if not local.exists():
            staged.append((pending, local))
        print(f'[{index}/{len(candidate["items"])}] {item["image_id"]} {state}', flush=True)
    duplicates = [ids for ids in checksums.values() if len(ids) > 1]
    if duplicates:
        raise ValueError(f"Doublons exacts dans la sélection candidate : {duplicates}")
    return staged


def activate(plan, active, candidate, staged):
    """Archiver les fichiers retirés ; revenir au jeu précédent si l'activation échoue."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = prep.DATA / "history" / stamp
    archive.mkdir(parents=True)
    state_files = ["selection_plan.json", "download_manifest.json", "metadata.csv",
                   "preparation_status.json", "apercu.jpg"]
    for name in state_files:
        if (prep.DATA / name).exists():
            shutil.copy2(prep.DATA / name, archive / name)
    keep = {item["filename"] for item in candidate["items"]}
    moved_out, moved_in = [], []
    try:
        for item in active["items"]:
            if item["filename"] in keep:
                continue
            path = prep.DATA / item["filename"]
            if path.exists():
                saved = archive / item["filename"]
                saved.parent.mkdir(parents=True, exist_ok=True)
                path.replace(saved)
                moved_out.append((saved, path))
        for pending, local in staged:
            if local.exists():
                raise ValueError(f"Un fichier est apparu pendant la mise à jour : {local}")
            local.parent.mkdir(parents=True, exist_ok=True)
            pending.replace(local)
            moved_in.append((local, pending))
        prep.write_json(prep.DATA / "download_manifest.json", candidate)
        prep.write_json(prep.DATA / "selection_plan.json", plan)
        report = prep.save_report(candidate)
        if report["status"] != "telecharge_a_revoir":
            raise ValueError("La vérification finale du jeu équilibré a échoué")
    except Exception:
        for local, pending in reversed(moved_in):
            pending.parent.mkdir(parents=True, exist_ok=True)
            local.replace(pending)
        for saved, path in reversed(moved_out):
            path.parent.mkdir(parents=True, exist_ok=True)
            saved.replace(path)
        for name in state_files:
            if (archive / name).exists():
                shutil.copy2(archive / name, prep.DATA / name)
        raise
    return archive


def rebalance(plan):
    active = load(prep.DATA / "download_manifest.json")
    prep.validate_items(active["items"])
    if load(prep.DATA / "selection_plan.json") == plan:
        report = prep.save_report(active)
        if report["status"] != "telecharge_a_revoir":
            raise ValueError("Le jeu actif doit être réparé avec prepare_data.py")
        print("Le jeu équilibré est déjà actif.")
        return {"status": "applique", "available_images": report["available_images"]}
    candidate = candidate_manifest(plan, active)
    for source_id in candidate["commits"]:
        if not (prep.DATA / "licenses" / f"{source_id}.txt").is_file():
            raise ValueError(f"Licence manquante : {source_id}. Relancer prepare_data.py avant la mise à jour.")
    staged = stage_downloads(candidate)
    archive = activate(plan, active, candidate, staged)
    return {"status": "applique", "available_images": len(candidate["items"]),
            "archive": str(archive.relative_to(prep.DATA)), "new_downloads": len(staged)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Afficher la cible sans téléchargement")
    args = parser.parse_args()
    plan = load(prep.DATA / "selection_plan_balanced.json")
    target = statistics(plan)
    if args.plan:
        print(json.dumps(target, ensure_ascii=False, indent=2))
        return 0
    status = {"updated_at": datetime.now(timezone.utc).isoformat(), **target}
    try:
        import PIL  # noqa: F401
        status.update(rebalance(plan))
    except Exception as exc:
        status.update(status="non_applique", error=str(exc))
        prep.write_json(prep.DATA / "rebalance_status.json", status)
        print(f"Rééquilibrage non appliqué : {exc}", file=sys.stderr)
        print("Le jeu actif est conservé. Relancer la même commande depuis un terminal connecté.", file=sys.stderr)
        return 1
    prep.write_json(prep.DATA / "rebalance_status.json", status)
    print("Rééquilibrage appliqué. Inventaire et aperçu mis à jour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
