"""Régressions réseau et préparation des données, sans serveur externe."""

import contextlib
import csv
import hashlib
import http.client
import io
import json
import tempfile
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import prepare_data as preparation


class Response:
    def __init__(self, payload, *, maximum_chunk=97, declared_length=None):
        self.stream = io.BytesIO(payload)
        self.maximum_chunk = maximum_chunk
        self.headers = {"Content-Length": str(len(payload) if declared_length is None else declared_length)}

    def read(self, size):
        return self.stream.read(min(size, self.maximum_chunk))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TransportTests(unittest.TestCase):
    url = "https://api.github.com/repos/example/data/git/ref/heads/main"

    def setUp(self):
        self.sleep = patch.object(preparation.time, "sleep").start()
        self.addCleanup(patch.stopall)
        self.stderr = contextlib.redirect_stderr(io.StringIO())
        self.stderr.__enter__()
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def test_reads_all_chunks_before_decoding_json(self):
        payload = b'{"object": {"sha": "complete-value"}}'
        with patch.object(preparation.urllib.request, "urlopen", return_value=Response(payload, maximum_chunk=2)) as request:
            self.assertEqual(preparation.fetch(self.url, parse_json=True)["object"]["sha"], "complete-value")
        self.assertEqual(request.call_count, 1)

    def test_retries_truncated_json_even_without_content_length(self):
        broken = Response(b'{"sha": "unfinished')
        broken.headers = {}
        with patch.object(preparation.urllib.request, "urlopen", side_effect=[broken, Response(b'{"sha": "ok"}')]) as request:
            self.assertEqual(preparation.fetch(self.url, parse_json=True), {"sha": "ok"})
        self.assertEqual(request.call_count, 2)

    def test_retries_when_content_length_does_not_match(self):
        with patch.object(preparation.urllib.request, "urlopen", side_effect=[
            Response(b'{"sha":', declared_length=100), Response(b'{"sha": "ok"}'),
        ]) as request:
            self.assertEqual(preparation.fetch(self.url, parse_json=True), {"sha": "ok"})
        self.assertEqual(request.call_count, 2)

    def test_retries_incomplete_read_exception(self):
        with patch.object(preparation.urllib.request, "urlopen", side_effect=[
            http.client.IncompleteRead(b"partial", 10), Response(b"complete"),
        ]) as request:
            self.assertEqual(preparation.fetch(self.url), b"complete")
        self.assertEqual(request.call_count, 2)

    def test_json_error_after_three_attempts_identifies_url(self):
        with patch.object(preparation.urllib.request, "urlopen", side_effect=lambda *args, **kwargs: Response(b'{"sha": "unfinished')) as request:
            with self.assertRaisesRegex(RuntimeError, "Réponse JSON invalide ou incomplète") as error:
                preparation.fetch(self.url, parse_json=True)
        self.assertIn(self.url, str(error.exception))
        self.assertEqual(request.call_count, 3)

    def test_404_is_not_retried(self):
        with patch.object(preparation.urllib.request, "urlopen", side_effect=urllib.error.HTTPError(
            self.url, 404, "Not Found", {}, None,
        )) as request:
            with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
                preparation.fetch(self.url)
        self.assertEqual(request.call_count, 1)

    def test_size_limit_is_preserved(self):
        with patch.object(preparation, "LIMIT", 4), patch.object(
            preparation.urllib.request, "urlopen", return_value=Response(b"12345"),
        ):
            with self.assertRaisesRegex(ValueError, "trop volumineux"):
                preparation.fetch(self.url)

    def test_refuses_truncated_git_tree(self):
        with patch.object(preparation, "api", return_value={"tree": [], "truncated": True}):
            with self.assertRaisesRegex(ValueError, "Liste de fichiers tronquée"):
                preparation.directory_files("example/data", "a" * 40, "Training", {})


class PreparationIntegrationTests(unittest.TestCase):
    def test_full_selection_and_resume_with_compact_api(self):
        plan = json.loads((preparation.DATA / "selection_plan.json").read_text())
        api_payloads, image_payloads, trees = {}, {}, {}

        def digest(text):
            return hashlib.sha1(text.encode()).hexdigest()

        def tree_sha(source_id, directory):
            return digest(source_id + "/" + directory)

        def put_json(url, value):
            api_payloads[url] = json.dumps(value).encode()

        for source_id, source in plan["sources"].items():
            repository = source["repository"]
            commit = digest(repository + "/commit")
            base = f"https://api.github.com/repos/{repository}"
            put_json(base + "/git/ref/heads/" + source["ref"], {"object": {"sha": commit, "type": "commit"}})
            put_json(base + "/git/commits/" + commit, {"tree": {"sha": tree_sha(source_id, "")}})
            image_payloads[preparation.raw_url(repository, commit, "LICENSE")] = b"Test licence fixture"
            trees[source_id, ""] = {}

        for group in plan["groups"]:
            source_id, directory = group["source"], group["directory"]
            source = plan["sources"][source_id]
            commit = digest(source["repository"] + "/commit")
            prefix = ""
            for part in directory.split("/"):
                child = f"{prefix}/{part}".lstrip("/")
                trees.setdefault((source_id, prefix), {})[part] = {
                    "path": part, "type": "tree", "mode": "040000", "sha": tree_sha(source_id, child),
                }
                trees.setdefault((source_id, child), {})
                prefix = child
            prefix = "pepper_" if "filename_pattern" in group else ""
            names = group.get("files", [f"{prefix}{index}_100.jpg" for index in range(group["count"] + 2)])
            for name in names:
                path = f"{directory}/{name}"
                token = hashlib.sha256((source_id + path).encode()).digest()
                stream = io.BytesIO()
                Image.new("RGB", (100, 100), tuple(token[:3])).save(stream, format="JPEG")
                payload = stream.getvalue()
                blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
                trees[source_id, directory][name] = {"path": name, "type": "blob", "mode": "100644", "sha": blob}
                image_payloads[preparation.raw_url(source["repository"], commit, path)] = payload

        for (source_id, directory), entries in trees.items():
            repository = plan["sources"][source_id]["repository"]
            put_json(f"https://api.github.com/repos/{repository}/git/trees/{tree_sha(source_id, directory)}", {
                "tree": list(entries.values()), "truncated": False,
            })

        def respond(request, timeout):
            url = request.full_url
            self.assertNotIn("/contents/", url)
            self.assertNotIn("/commits/main", url)
            self.assertNotIn("recursive=", url)
            payload = api_payloads[url] if url in api_payloads else image_payloads[url]
            return Response(payload)

        with tempfile.TemporaryDirectory(prefix="fruit-preparation-test-") as temporary, patch.object(
            preparation, "DATA", Path(temporary),
        ), patch.object(preparation.urllib.request, "urlopen", side_effect=respond) as requests, contextlib.redirect_stdout(io.StringIO()):
            preparation.write_json(preparation.DATA / "selection_plan.json", plan)
            with patch.object(preparation.sys, "argv", ["prepare_data.py"]):
                self.assertEqual(preparation.main(), 0)
                status = json.loads((preparation.DATA / "preparation_status.json").read_text())
                expected_total = sum(group["count"] for group in plan["groups"])
                expected_split, expected_source = Counter(), Counter()
                for group in plan["groups"]:
                    expected_split[group["split"]] += group["count"]
                    expected_source[group["source"]] += group["count"]
                self.assertEqual(status["available_images"], expected_total)
                self.assertEqual(status["by_split"], expected_split)
                with (preparation.DATA / "metadata.csv").open() as stream:
                    records = list(csv.DictReader(stream))
                self.assertEqual(len(records), expected_total)
                self.assertEqual(Counter(record["source"] for record in records), expected_source)
                self.assertTrue((preparation.DATA / "apercu.jpg").is_file())
                requests.reset_mock()
                self.assertEqual(preparation.main(), 0)
                requests.assert_not_called()
            with patch.object(preparation.sys, "argv", ["prepare_data.py", "--check"]):
                self.assertEqual(preparation.main(), 0)
                requests.assert_not_called()


if __name__ == "__main__":
    unittest.main()
