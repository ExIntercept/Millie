"""Tests for Workspace File Suite, Knowledge Base, and App Opener."""
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from backend import apps, config, files
from backend.db import DB
from backend.tools import ToolBox
from backend.hub import Hub


class TestWorkspaceSuite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.orig_workspace = config.WORKSPACE_DIR
        self.orig_db_path = config.DB_PATH
        config.WORKSPACE_DIR = Path(self.tmp.name) / "workspace"
        config.DB_PATH = Path(self.tmp.name) / "test.db"
        self.db = DB(config.DB_PATH)
        self.hub = Hub()
        self.toolbox = ToolBox(self.db, self.hub, lambda tid, instr: True)

    def tearDown(self):
        config.WORKSPACE_DIR = self.orig_workspace
        config.DB_PATH = self.orig_db_path
        self.tmp.cleanup()

    def test_create_and_read_docx(self):
        content = """# Executive Summary
This is a test docx file generated from markdown.

## Key Highlights
- First item
- Second item with details

| Metric | Value |
| --- | --- |
| Accuracy | 98% |
| Speed | 120ms |
"""
        target = files.create_workspace_file("test_report.docx", content)
        self.assertTrue(target.exists())
        self.assertGreater(target.stat().st_size, 1000)

        # Read back
        res = files.read_workspace_file("test_report.docx")
        self.assertEqual(res["name"], "test_report.docx")
        self.assertIn("Executive Summary", res["content"])
        self.assertIn("Key Highlights", res["content"])
        self.assertIn("First item", res["content"])

    def test_file_conversions(self):
        md_text = "# Sample Document\n\n- Bullet A\n- Bullet B\n\n```python\nprint('hello world')\n```\n\nSpecial <chars> & symbols."
        files.create_workspace_file("doc.md", md_text)

        # 1. Convert to .docx
        c_docx = files.convert_file("doc.md", ".docx", "doc_converted")
        self.assertTrue(c_docx.exists())
        self.assertEqual(c_docx.suffix, ".docx")
        self.assertGreater(c_docx.stat().st_size, 1000)

        # 2. Convert to .html
        c_html = files.convert_file("doc.md", ".html")
        self.assertTrue(c_html.exists())
        html_text = c_html.read_text()
        self.assertIn("<h1", html_text)
        self.assertIn("Bullet A", html_text)

        # 3. Convert to .txt
        c_txt = files.convert_file("doc.md", ".txt")
        self.assertTrue(c_txt.exists())
        self.assertIn("Sample Document", c_txt.read_text())

        # 4. Convert to .pdf (tests ReportLab XML escaping and formatting)
        c_pdf = files.convert_file("doc.md", ".pdf")
        self.assertTrue(c_pdf.exists())
        self.assertEqual(c_pdf.suffix, ".pdf")
        self.assertGreater(c_pdf.stat().st_size, 1000)

        # 5. Convert .docx to .pdf
        c_docx_pdf = files.convert_file("doc_converted.docx", ".pdf", "docx_as_pdf")
        self.assertTrue(c_docx_pdf.exists())
        self.assertEqual(c_docx_pdf.suffix, ".pdf")
        self.assertGreater(c_docx_pdf.stat().st_size, 1000)

        # 6. Convert image to .pdf
        from PIL import Image
        img_path = config.WORKSPACE_DIR / "sample.png"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(img_path)
        c_img_pdf = files.convert_file("sample.png", ".pdf")
        self.assertTrue(c_img_pdf.exists())
        self.assertEqual(c_img_pdf.suffix, ".pdf")
        self.assertGreater(c_img_pdf.stat().st_size, 500)

    def test_merge_files(self):
        # 1. Merge Markdown files
        files.create_workspace_file("p1.md", "# Part 1\nIntro text")
        files.create_workspace_file("p2.md", "# Part 2\nConclusion text")
        merged_md = files.merge_files(["p1.md", "p2.md"], "combined.md")
        self.assertTrue(merged_md.exists())
        text = merged_md.read_text()
        self.assertIn("Part 1", text)
        self.assertIn("Part 2", text)

        # 2. Merge Word Documents (.docx)
        files.create_workspace_file("w1.docx", "# Section One\nFirst body paragraph.")
        files.create_workspace_file("w2.docx", "# Section Two\nSecond body paragraph.")
        merged_docx = files.merge_files(["w1.docx", "w2.docx"], "combined_word.docx")
        self.assertTrue(merged_docx.exists())
        self.assertGreater(merged_docx.stat().st_size, 2000)
        # Read back merged docx
        read_back = files.read_workspace_file("combined_word.docx")
        self.assertIn("Section One", read_back["content"])
        self.assertIn("Section Two", read_back["content"])

        # 3. Merge Mixed Formats into Word Document (.docx)
        merged_mixed_docx = files.merge_files(["p1.md", "w1.docx"], "mixed_word.docx")
        self.assertTrue(merged_mixed_docx.exists())
        read_mixed = files.read_workspace_file("mixed_word.docx")
        self.assertIn("Part 1", read_mixed["content"])

        # 4. Merge PDFs and mixed files into single PDF
        c_pdf1 = files.convert_file("p1.md", ".pdf", "pdf1.pdf")
        c_pdf2 = files.convert_file("p2.md", ".pdf", "pdf2.pdf")
        merged_pdf = files.merge_files(["pdf1.pdf", "pdf2.pdf"], "combined.pdf")
        self.assertTrue(merged_pdf.exists())
        self.assertGreater(merged_pdf.stat().st_size, 1000)

        # 5. Merge mixed PDF + TXT into PDF
        files.create_workspace_file("extra.txt", "Some extra plain text notes.")
        merged_hybrid_pdf = files.merge_files(["pdf1.pdf", "extra.txt"], "hybrid.pdf")
        self.assertTrue(merged_hybrid_pdf.exists())
        self.assertGreater(merged_hybrid_pdf.stat().st_size, 1000)

    def test_knowledge_base_crud_and_fts(self):
        # Add facts
        k1 = self.db.add_knowledge(
            "User prefers Python 3.11",
            "Always use python 3.11 for scripts and venvs on macOS.",
            category="preference",
            tags=["python", "mac"]
        )
        k2 = self.db.add_knowledge(
            "MedGemma Project Overview",
            "MedGemma is a local clinical AI pipeline using Whisper and Groq.",
            category="project",
            tags=["clinical", "groq"]
        )
        self.assertIsInstance(k1, int)
        self.assertIsInstance(k2, int)

        # List
        all_k = self.db.list_knowledge()
        self.assertEqual(len(all_k), 2)

        # Search FTS
        hits = self.db.search_knowledge("clinical Whisper")
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["title"], "MedGemma Project Overview")

        # Update
        self.db.update_knowledge(k1, title="User strictly prefers Python 3.11")
        updated = self.db.get_knowledge(k1)
        self.assertEqual(updated["title"], "User strictly prefers Python 3.11")

        # Delete
        self.assertTrue(self.db.delete_knowledge(k1))
        self.assertIsNone(self.db.get_knowledge(k1))

    def test_app_opener_alias(self):
        self.assertEqual(apps.APP_ALIASES["chrome"], "Google Chrome")
        self.assertEqual(apps.APP_ALIASES["vscode"], "Visual Studio Code")
        self.assertEqual(apps.APP_ALIASES["spotify"], "Spotify")

    @mock.patch("backend.apps.subprocess.run")
    def test_browse_lookup_formatting(self, run):
        # subprocess.run is faked so the test doesn't open real browser tabs
        run.return_value = mock.Mock(returncode=0, stderr="")
        # 1. Topic search query
        res = apps.browse_lookup("quantum computing")
        self.assertIn("google.com/search?q=quantum+computing", res["url"])

        # 2. YouTube search
        res_yt = apps.browse_lookup("lofi chill beats", engine="youtube")
        self.assertIn("youtube.com/results?search_query=lofi+chill+beats", res_yt["url"])

        # 3. Direct domain / URL
        res_url = apps.browse_lookup("github.com/trending")
        self.assertEqual(res_url["url"], "https://github.com/trending")

    def test_workspace_api_endpoints(self):
        from starlette.testclient import TestClient
        from backend.main import app

        client = TestClient(app)

        # 1. Upload file
        file_payload = ("test_upload.md", b"# Uploaded Title\nUploaded content body.", "text/markdown")
        res = client.post("/api/workspace/upload", files={"file": file_payload})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])
        self.assertEqual(res.json()["rel_path"], "test_upload.md")

        # 2. Read back uploaded file content
        res = client.get("/api/workspace/files/content", params={"path": "test_upload.md"})
        self.assertEqual(res.status_code, 200)
        self.assertIn("Uploaded Title", res.json()["content"])

        # 3. Convert via endpoint
        res = client.post("/api/workspace/convert", json={
            "src_path": "test_upload.md",
            "target_ext": ".docx",
            "out_name": "uploaded_as_docx.docx"
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])

        # 4. Upload second file to test merge via endpoint
        res = client.post("/api/workspace/upload", files={
            "file": ("second_upload.md", b"# Second Part\nSecond body.", "text/markdown")
        })
        self.assertEqual(res.status_code, 200)

        # 5. Merge via endpoint
        res = client.post("/api/workspace/merge", json={
            "src_paths": ["test_upload.md", "second_upload.md"],
            "out_name": "api_merged.md"
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])

        # Verify merged content
        res = client.get("/api/workspace/files/content", params={"path": "api_merged.md"})
        self.assertEqual(res.status_code, 200)
        self.assertIn("Uploaded Title", res.json()["content"])
        self.assertIn("Second Part", res.json()["content"])


if __name__ == "__main__":
    unittest.main()

