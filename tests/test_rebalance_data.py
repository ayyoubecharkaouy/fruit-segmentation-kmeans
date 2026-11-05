"""Mise à jour du jeu actif et reprise sans perte de données, entièrement hors réseau."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import prepare_data as prep
from scripts import rebalance_data as rebalance


def fixture_responses(old_plan, target):
    responses, trees = {}, {}

    def digest(text):
        return hashlib.sha1(text.encode()).hexdigest()

    def put_json(url, value):
        responses[url] = json.dumps(value).encode()

    for source_id, source in target["sources"].items():
        repository = source["repository"]
        commit = digest(repository + "/commit")
        root = digest(source_id + "/")
        base = f"https://api.github.com/repos/{repository}"
        put_json(base + "/git/ref/heads/" + source["ref"], {"object": {"sha": commit}})
        put_json(base + "/git/commits/" + commit, {"tree": {"sha": root}})
        responses[prep.raw_url(repository, commit, "LICENSE")] = b"Test licence"
        trees[source_id, ""] = {}
    for group in old_plan["groups"] + target["groups"]:
        source_id, directory = group["source"], group["directory"]
        source = target["sources"][source_id]
        prefix = ""
        for part in directory.split("/"):
            child = f"{prefix}/{part}".lstrip("/")
            trees.setdefault((source_id, prefix), {})[part] = {
                "path": part, "type": "tree", "sha": digest(source_id + "/" + child),
            }
            trees.setdefault((source_id, child), {})
            prefix = child
        names = group.get("files", [
            ("pepper_" if "filename_pattern" in group else "") + f"{index}_100.jpg"
            for index in range(group["count"] + 2)
        ])
        for name in names:
            path = directory + "/" + name
            color = hashlib.sha256((source_id + path).encode()).digest()[:3]
            stream = io.BytesIO()
            Image.new("RGB", (40, 40), tuple(color)).save(stream, format="JPEG")
            payload = stream.getvalue()
            sha = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            trees[source_id, directory][name] = {"path": name, "type": "blob", "sha": sha}
            responses[prep.raw_url(source["repository"], digest(source["repository"] + "/commit"), path)] = payload
    for (source_id, directory), entries in trees.items():
        repository = target["sources"][source_id]["repository"]
        put_json(f"https://api.github.com/repos/{repository}/git/trees/{digest(source_id + '/' + directory)}", {
            "tree": list(entries.values()), "truncated": False,
        })
    return responses


class RebalanceTests(unittest.TestCase):
    def setUp(self):
        self.target = json.loads((prep.DATA / "selection_plan_balanced.json").read_text())
        # Le plan initial reste une fixture de 45 images même après activation locale de la v2.
        self.old_plan = {**self.target, "groups": [group for group in self.target["groups"]
                                                 if group["source"] == "fruits360_simple"]}
        common = {"source": "fruits360_multi", "directory": "test-multiple_fruits",
                  "background": "a_verifier", "difficulty": "a_verifier"}
        self.old_plan["groups"] += [
            {**common, "id": "pommes_multi", "count": 10,
             "files": [f"apple_{index}.jpg" for index in range(1, 11)],
             "fruit_types": "pomme", "split": "mise_au_point"},
            {**common, "id": "bananes_multi", "scene_id": "banane_lady_finger", "count": 5,
             "files": [f"Banana(lady_finger)_{index}.jpg" for index in range(1, 6)],
             "fruit_types": "banane", "split": "evaluation"},
        ]
        self.responses = fixture_responses(self.old_plan, self.target)
        self.temporary = tempfile.TemporaryDirectory(prefix="rebalance-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(prep, "DATA", self.directory))
        self.patches.enter_context(patch.object(prep.time, "sleep"))
        self.patches.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.patches.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.failed_name = None
        self.requests = self.patches.enter_context(patch.object(prep.urllib.request, "urlopen", side_effect=self.respond))
        prep.write_json(self.directory / "selection_plan.json", self.old_plan)
        prep.write_json(self.directory / "selection_plan_balanced.json", self.target)
        with patch.object(prep.sys, "argv", ["prepare_data.py"]):
            self.assertEqual(prep.main(), 0)
        self.active = rebalance.load(self.directory / "download_manifest.json")
        self.before = {name: (self.directory / name).read_bytes() for name in (
            "selection_plan.json", "download_manifest.json", "metadata.csv", "preparation_status.json", "apercu.jpg",
        )}
        self.requests.reset_mock()

    def respond(self, request, timeout):
        url = request.full_url
        if self.failed_name and self.failed_name in url:
            raise urllib.error.URLError("simulated interrupted download")
        response = io.BytesIO(self.responses[url])
        response.headers = {"Content-Length": str(len(self.responses[url]))}
        return response

    def run_rebalance(self):
        with patch.object(prep.sys, "argv", ["rebalance_data.py"]):
            return rebalance.main()

    def assert_original_intact(self):
        for name, content in self.before.items():
            self.assertEqual((self.directory / name).read_bytes(), content, name)
        self.assertEqual(len(list((self.directory / "raw").rglob("*.jpg"))), 45)

    def test_balanced_activation_reuses_40_downloads_8_and_archives_5(self):
        self.assertEqual(self.run_rebalance(), 0)
        active = rebalance.load(self.directory / "download_manifest.json")
        self.assertEqual(len(active["items"]), 48)
        multi = [item for item in active["items"] if item["source"] == "fruits360_multi"]
        counts = Counter(item["fruit_types"] for item in multi)
        self.assertEqual(counts["pomme"], 5)
        self.assertEqual(counts["banane"], 5)
        self.assertEqual(counts["poivron"], 5)
        self.assertEqual(sum(count for label, count in counts.items() if "|" in label), 3)
        previous_by_source = {(item["source"], item["source_file"]): item for item in self.active["items"]}
        kept = 0
        for item in active["items"]:
            previous = previous_by_source.get((item["source"], item["source_file"]))
            if previous:
                kept += 1
                self.assertEqual(item["image_id"], previous["image_id"])
                self.assertEqual(item["filename"], previous["filename"])
        self.assertEqual(kept, 40)
        downloads = [call.args[0].full_url for call in self.requests.call_args_list
                     if call.args[0].full_url.startswith("https://raw.githubusercontent.com/")]
        self.assertEqual(len(downloads), 8)
        status = rebalance.load(self.directory / "preparation_status.json")
        self.assertEqual(status["available_images"], 48)
        self.assertEqual(status["by_split"], {"mise_au_point": 33, "evaluation": 15})
        self.assertEqual(len(list((self.directory / "raw" / "fruits360_multi").glob("*.jpg"))), 18)
        self.assertEqual(len(list((self.directory / "history").rglob("raw/fruits360_multi/*.jpg"))), 5)
        self.requests.reset_mock()
        self.assertEqual(self.run_rebalance(), 0)
        self.requests.assert_not_called()

    def test_failed_download_keeps_active_dataset_and_resumes_from_staging(self):
        candidate = rebalance.candidate_manifest(self.target, self.active)
        old_paths = {item["filename"] for item in self.active["items"]}
        new_items = [item for item in candidate["items"] if item["filename"] not in old_paths]
        self.failed_name = new_items[1]["source_url"].rsplit("/", 1)[1]
        self.assertEqual(self.run_rebalance(), 1)
        self.assert_original_intact()
        self.assertTrue((self.directory / "staging" / "rebalance" / new_items[0]["filename"]).is_file())
        self.failed_name = None
        self.requests.reset_mock()
        self.assertEqual(self.run_rebalance(), 0)
        self.assertEqual(self.requests.call_count, 7)

    def test_activation_error_restores_original_files_and_inventory(self):
        with patch.object(prep, "save_report", side_effect=RuntimeError("simulated write error")):
            self.assertEqual(self.run_rebalance(), 1)
        self.assert_original_intact()
        self.assertEqual(self.run_rebalance(), 0)

    def test_missing_pepper_quota_does_not_change_active_files(self):
        original = prep.directory_files
        def no_peppers(repository, tree_sha, directory, cache):
            entries = original(repository, tree_sha, directory, cache)
            return [entry for entry in entries if not entry["name"].startswith("pepper_")]
        with patch.object(prep, "directory_files", side_effect=no_peppers):
            self.assertEqual(self.run_rebalance(), 1)
        status = rebalance.load(self.directory / "rebalance_status.json")
        self.assertIn("poivrons_multi", status["error"])
        self.assert_original_intact()


if __name__ == "__main__":
    unittest.main()
