"""Imports locaux, noms uniques et conservation des références, sans accès réseau."""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import organize_data as organization
from scripts import prepare_data as prep


def image_payload(format, color):
    stream = io.BytesIO()
    Image.new("RGB", (40, 40), color).save(stream, format=format)
    return stream.getvalue()


class OrganizationTests(unittest.TestCase):
    def setUp(self):
        template = json.loads((prep.DATA / "manual_additions.json").read_text())
        self.temporary = tempfile.TemporaryDirectory(prefix="organization-test-")
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(prep, "DATA", self.data))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        items = []
        for number, color in [(31, (200, 50, 20)), (32, (50, 200, 20))]:
            payload = image_payload("JPEG", color)
            path = self.data / f"raw/fruits360_multi/img{number:03d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            items.append({
                "image_id": f"img{number:03d}", "filename": str(path.relative_to(self.data)),
                "source": "fruits360_multi", "fruit_types": "pomme", "background": "etal",
                "lighting": "clair", "difficulty": "moyenne", "scene_id": "apples_multi",
                "split": "mise_au_point", "source_file": f"test-multiple_fruits/apple_{number}.jpg",
                "source_url": "https://example.invalid/original.jpg", "commit": "a" * 40,
                "license": "CC-BY-SA-4.0", "annotation_status": "non_annote",
                "review_status": "a_verifier_visuellement",
                "git_blob_sha": hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest(),
            })
        self.original_manifest = {"seed": 42, "commits": {"fruits360_multi": "a" * 40}, "items": items}
        prep.write_json(self.data / "download_manifest.json", self.original_manifest)
        plan = {"seed": 42, "sources": {}, "groups": [{"count": 2, "split": "mise_au_point"}]}
        for name in ["selection_plan.json", "selection_plan_balanced.json"]:
            prep.write_json(self.data / name, plan)
        (self.data / "licenses").mkdir()
        (self.data / "licenses/fruits360_multi.txt").write_text("Test licence")
        prep.save_report(self.original_manifest)
        for index, item in enumerate(template["items"]):
            format = "PNG" if item["filename"].endswith(".png") else "JPEG"
            payload = image_payload(format, (20 + index * 20, 30, 200))
            path = self.data / item["input_filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            item["sha256"] = hashlib.sha256(payload).hexdigest()
        prep.write_json(self.data / "manual_additions.json", template)
        self.additions = template["items"]
        self.original_state = {name: (self.data / name).read_bytes() for name in (
            "download_manifest.json", "selection_plan.json", "selection_plan_balanced.json",
            "metadata.csv", "preparation_status.json", "apercu.jpg",
        )}

    def test_import_renames_unique_ids_preserves_sources_and_bytes(self):
        with patch.object(prep, "fetch") as network:
            organization.organize()
            network.assert_not_called()
        manifest = organization.load(self.data / "download_manifest.json")
        self.assertEqual(len(manifest["items"]), 5)
        self.assertEqual({item["image_id"] for item in manifest["items"]}, {"img031", "img032", "img054", "img055", "img056"})
        for item in self.additions:
            self.assertFalse((self.data / item["input_filename"]).exists())
            payload = (self.data / item["filename"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
        with (self.data / "metadata.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 5)
        self.assertEqual(next(row for row in rows if row["image_id"] == "img055")["dataset_title"], "Chili Plant Disease Detection")
        report = organization.load(self.data / "preparation_status.json")
        self.assertEqual(report["untracked_images"], [])
        self.assertEqual(set(report["provenance_pending"]), {"img054", "img055", "img056"})
        with patch.object(prep, "fetch") as network, patch.object(prep.sys, "argv", ["prepare_data.py", "--check"]):
            self.assertEqual(prep.main(), 0)
            network.assert_not_called()
        count = len(list((self.data / "history").iterdir()))
        organization.organize()
        self.assertEqual(len(list((self.data / "history").iterdir())), count)

    def test_failure_restores_filenames_and_manifests(self):
        with patch.object(prep, "save_report", side_effect=RuntimeError("simulated write error")):
            with self.assertRaisesRegex(RuntimeError, "simulated write error"):
                organization.organize()
        for name, payload in self.original_state.items():
            self.assertEqual((self.data / name).read_bytes(), payload)
        for item in self.additions:
            self.assertTrue((self.data / item["input_filename"]).exists())
            self.assertFalse((self.data / item["filename"]).exists())

    def test_refuses_content_extension_mismatch(self):
        config = organization.load(self.data / "manual_additions.json")
        config["items"][0]["filename"] = config["items"][0]["filename"].replace(".png", ".jpg")
        prep.write_json(self.data / "manual_additions.json", config)
        with self.assertRaisesRegex(ValueError, "Extension incohérente"):
            organization.organize()

    def test_detects_corruption_of_local_import_without_network(self):
        organization.organize()
        item = self.additions[0]
        (self.data / item["filename"]).write_bytes(b"corrupted")
        manifest = organization.load(self.data / "download_manifest.json")
        _, _, invalid, _ = prep.audit(manifest)
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]["image_id"], item["image_id"])


if __name__ == "__main__":
    unittest.main()
