"""Turn uploaded files into text so each research tab can keep context cheaply.

Images are only OCR'd if the user switched the OCR toggle on; the text (not the
image) is what gets used as context.
"""
import logging
from pathlib import Path

from . import config

log = logging.getLogger("millie.files")

TEXT_EXT = {".txt", ".md", ".csv", ".json", ".py", ".tex", ".log", ".yaml", ".yml"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".heic"}
MAX_CHARS = 200_000


def classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in TEXT_EXT or ext in {".pdf", ".pptx", ".docx"}:
        return "text"
    return "other"


def ocr_image(path: Path) -> str:
    """Try Apple Vision (macOS), then Tesseract. Raises RuntimeError if neither exists."""
    try:
        from ocrmac import ocrmac  # Apple Vision framework, fast on M-series
        results = ocrmac.OCR(str(path)).recognize()
        return "\n".join(r[0] for r in results)
    except ImportError:
        pass
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(path))
    except ImportError:
        pass
    raise RuntimeError("No OCR engine installed. Run: pip install ocrmac (macOS) or pytesseract")


def _pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join(f"[page {i + 1}]\n{(p.extract_text() or '').strip()}" for i, p in enumerate(reader.pages))


def _pptx(path: Path) -> str:
    from pptx import Presentation
    out = []
    for i, slide in enumerate(Presentation(str(path)).slides, 1):
        parts = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    parts.append(" | ".join(c.text for c in row.cells))
        if slide.has_notes_slide:
            parts.append("(notes) " + slide.notes_slide.notes_text_frame.text)
        out.append(f"[slide {i}]\n" + "\n".join(p for p in parts if p.strip()))
    return "\n\n".join(out)


def _docx(path: Path) -> str:
    import docx
    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def extract(path: Path, ocr: bool) -> tuple[str, str]:
    """Return (kind, text). Never raises: problems become a short note in the text."""
    kind = classify(path)
    ext = path.suffix.lower()
    try:
        if kind == "image":
            text = ocr_image(path) if ocr else ""
        elif ext == ".pdf":
            text = _pdf(path)
        elif ext == ".pptx":
            text = _pptx(path)
        elif ext == ".docx":
            text = _docx(path)
        elif kind == "text":
            text = path.read_text(errors="replace")
        else:
            text = ""
    except Exception as e:  # missing library, corrupt file, ...
        log.warning("extract failed for %s: %s", path.name, e)
        text = f"[could not read {path.name}: {e}]"
    return kind, text[:MAX_CHARS]


# -----------------------------------------------------------------------------
# Workspace Storage, Conversion & Merging Suite
# -----------------------------------------------------------------------------


def get_workspace_dir() -> Path:
    config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return config.WORKSPACE_DIR.resolve()


def _safe_rel(rel_path: str) -> Path:
    base = get_workspace_dir().resolve()
    target = (base / rel_path.lstrip("/")).resolve()
    if not target.is_relative_to(base):   # a prefix check would let "../workspace2" through
        raise ValueError(f"Path '{rel_path}' escapes workspace directory.")
    return target


def create_docx_from_markdown(dest: Path, markdown_text: str, title: str = "") -> Path:
    """Create a nicely formatted Word (.docx) document from markdown/plain text."""
    import docx
    doc = docx.Document()
    if title:
        doc.add_heading(title, 0)
    lines = markdown_text.split("\n")
    table_rows: list[list[str]] = []

    def flush_table():
        nonlocal table_rows
        if not table_rows:
            return
        ncols = max(len(r) for r in table_rows)
        table = doc.add_table(rows=len(table_rows), cols=ncols)
        table.style = "Table Grid"
        for r_idx, row in enumerate(table_rows):
            for c_idx, val in enumerate(row):
                if c_idx < ncols:
                    table.cell(r_idx, c_idx).text = val
        table_rows = []

    for line in lines:
        s = line.strip()
        if not s:
            flush_table()
            continue
        if s.startswith("# "):
            flush_table()
            doc.add_heading(s[2:], 1)
        elif s.startswith("## "):
            flush_table()
            doc.add_heading(s[3:], 2)
        elif s.startswith("### "):
            flush_table()
            doc.add_heading(s[4:], 3)
        elif s.startswith(("- ", "* ", "• ")):
            flush_table()
            doc.add_paragraph(s[2:], style="List Bullet")
        elif s.startswith("|") and s.endswith("|"):
            parts = [p.strip() for p in s.strip("|").split("|")]
            if all(set(p).issubset({"-", ":", " "}) for p in parts):
                continue
            table_rows.append(parts)
        else:
            flush_table()
            doc.add_paragraph(s)
    flush_table()
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(dest))
    return dest


def _is_mostly_ascii_art(lines: list[str]) -> bool:
    non_blank = [l for l in lines if l.strip()]
    if not non_blank:
        return False
    art_chars = set("┌┐└┘├┤┬┴┼─│═║╔╗╚╝╠╣╦╩╬░▒▓█▄▀▲▼◆○●★☆#/_\\|`~^")
    art_count = 0
    for l in non_blank:
        stripped = l.strip()
        if len(stripped) > 6 and sum(1 for c in stripped if c in art_chars) / len(stripped) > 0.45:
            art_count += 1
    return (art_count / len(non_blank)) > 0.5


def create_pdf_from_text(dest: Path, content: str, title: str = "") -> Path:
    """Generate a clean, valid PDF document from text, markdown, or ASCII art."""
    import html
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    dest.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(dest), pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    story = []

    if title:
        story.append(Paragraph(html.escape(title, quote=False), styles["Title"]))
        story.append(Spacer(1, 14))

    lines = content.split("\n")
    mono_style = ParagraphStyle("Mono", fontName="Courier", fontSize=9.0, leading=12)

    if _is_mostly_ascii_art(lines):
        story.append(Preformatted(content, mono_style))
    else:
        in_code = False
        code_chunk: list[str] = []

        for line in lines:
            if line.strip().startswith("```"):
                if in_code:
                    story.append(Preformatted("\n".join(code_chunk), mono_style))
                    story.append(Spacer(1, 6))
                    code_chunk = []
                    in_code = False
                else:
                    in_code = True
                continue

            if in_code:
                code_chunk.append(line)
                continue

            s = line.strip()
            if not s:
                story.append(Spacer(1, 6))
            elif s.startswith("# "):
                story.append(Paragraph(html.escape(s[2:], quote=False), styles["Heading1"]))
                story.append(Spacer(1, 6))
            elif s.startswith("## "):
                story.append(Paragraph(html.escape(s[3:], quote=False), styles["Heading2"]))
                story.append(Spacer(1, 4))
            elif s.startswith("### "):
                story.append(Paragraph(html.escape(s[4:], quote=False), styles["Heading3"]))
                story.append(Spacer(1, 4))
            elif s.startswith(("- ", "* ", "• ")):
                story.append(Paragraph(f"&bull; {html.escape(s[2:], quote=False)}", styles["BodyText"]))
            else:
                story.append(Paragraph(html.escape(s, quote=False), styles["BodyText"]))
                story.append(Spacer(1, 3))

        if in_code and code_chunk:
            story.append(Preformatted("\n".join(code_chunk), mono_style))

    doc.build(story)
    return dest


def create_workspace_file(rel_path: str, content: str) -> Path:
    """Create or overwrite a file in the workspace directory."""
    target = _safe_rel(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ext = target.suffix.lower()

    if ext == ".docx":
        title = target.stem.replace("_", " ").title()
        create_docx_from_markdown(target, content, title=title)
    elif ext == ".pdf":
        title = target.stem.replace("_", " ").title()
        create_pdf_from_text(target, content, title=title)
    elif ext in {".png", ".jpg", ".jpeg"} and content.startswith("data:image"):
        import base64
        data = content.split(",", 1)[-1]
        target.write_bytes(base64.b64decode(data))
    else:
        target.write_text(content, encoding="utf-8")
    return target



def read_workspace_file(rel_path: str) -> dict:
    """Read a workspace file and return preview text and metadata."""
    target = _safe_rel(rel_path)
    if not target.exists():
        raise FileNotFoundError(f"File '{rel_path}' not found in workspace.")
    stat = target.stat()
    kind, text = extract(target, ocr=False)
    return {
        "name": target.name,
        "rel_path": str(target.relative_to(get_workspace_dir())),
        "kind": kind,
        "ext": target.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
        "content": text[:50000],
    }


def delete_workspace_file(rel_path: str) -> bool:
    """Delete a workspace file or folder (never the workspace itself)."""
    target = _safe_rel(rel_path)
    if target == get_workspace_dir():
        raise ValueError("Refusing to delete the whole workspace.")
    if not target.exists():
        return False
    if target.is_dir():
        import shutil
        shutil.rmtree(target)
    else:
        target.unlink()
    return True


def list_workspace(rel_dir: str = "") -> dict:
    """List directory contents and hierarchical tree of the workspace."""
    base = get_workspace_dir()
    curr = _safe_rel(rel_dir) if rel_dir else base
    if not curr.exists() or not curr.is_dir():
        return {"files": [], "tree": []}

    def fmt_size(sz: int) -> str:
        if sz < 1024:
            return f"{sz} B"
        if sz < 1024 * 1024:
            return f"{sz / 1024:.1f} KB"
        return f"{sz / (1024 * 1024):.1f} MB"

    flat = []
    for p in sorted(curr.glob("**/*")):
        if p.name.startswith("."):
            continue
        rel = str(p.relative_to(base))
        stat = p.stat()
        flat.append({
            "name": p.name,
            "rel_path": rel,
            "is_dir": p.is_dir(),
            "kind": "directory" if p.is_dir() else classify(p),
            "ext": p.suffix.lower() if not p.is_dir() else "",
            "size_bytes": stat.st_size if not p.is_dir() else 0,
            "size_human": fmt_size(stat.st_size) if not p.is_dir() else "",
            "modified_at": stat.st_mtime,
        })
    return {"files": flat}


def convert_file(src_rel: str, target_ext: str, out_name: str = "") -> Path:
    """Convert an existing workspace file to target_ext (.docx, .md, .txt, .html, .pdf, or image formats)."""
    src = _safe_rel(src_rel)
    if not src.exists():
        raise FileNotFoundError(f"Source file '{src_rel}' not found.")
    target_ext = ("." + target_ext.lstrip(".")).lower()
    out_stem = out_name if out_name else f"{src.stem}_converted"
    if not out_stem.endswith(target_ext):
        out_stem += target_ext
    dest = src.parent / out_stem

    s_ext = src.suffix.lower()
    if s_ext == target_ext:
        import shutil
        shutil.copy2(src, dest)
        return dest

    # 1. To DOCX
    if target_ext == ".docx":
        if s_ext in IMAGE_EXT:
            import docx
            doc = docx.Document()
            doc.add_heading(src.stem.replace("_", " ").title(), 0)
            doc.add_picture(str(src))
            dest.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(dest))
            return dest
        _, text = extract(src, ocr=False)
        return create_docx_from_markdown(dest, text, title=src.stem.replace("_", " ").title())

    # 2. To Markdown / TXT / HTML
    if target_ext in {".md", ".txt", ".html"}:
        _, text = extract(src, ocr=(s_ext in IMAGE_EXT))
        if target_ext == ".html":
            import markdown
            html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])
            dest.write_text(
                f"<!doctype html><html><head><meta charset='utf-8'><title>{src.stem}</title>"
                f"<style>body{{font-family:system-ui,sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem;line-height:1.6;}}"
                f"table{{border-collapse:collapse;width:100%;margin:1rem 0;}}th,td{{border:1px solid #ddd;padding:8px;}}th{{background:#f4f4f4;}}</style>"
                f"</head><body>{html_body}</body></html>",
                encoding="utf-8",
            )
        else:
            dest.write_text(text, encoding="utf-8")
        return dest

    # 3. To PDF
    if target_ext == ".pdf":
        if s_ext in IMAGE_EXT:
            from PIL import Image
            img = Image.open(src)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dest)
            return dest
        else:
            _, text = extract(src, ocr=False)
            return create_pdf_from_text(dest, text, title=src.stem.replace("_", " ").title())

    # 4. Image conversions
    if target_ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} and s_ext in IMAGE_EXT:
        from PIL import Image
        img = Image.open(src)
        if target_ext in {".jpg", ".jpeg"} and img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dest)
        return dest

    raise ValueError(f"Conversion from '{s_ext}' to '{target_ext}' is not supported.")


def merge_files(src_rels: list[str], out_name: str) -> Path:
    """Merge multiple files (PDFs, Word documents, images, or text) into one file."""
    if not src_rels:
        raise ValueError("No files provided to merge.")
    paths = [_safe_rel(r) for r in src_rels]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"File '{p.name}' not found.")

    dest_name = out_name.strip()
    dest_ext = Path(dest_name).suffix.lower()

    # Case A: Merge into PDF
    if dest_ext == ".pdf" or (not dest_ext and all(p.suffix.lower() == ".pdf" for p in paths)):
        if not dest_name.lower().endswith(".pdf"):
            dest_name += ".pdf"
        dest = paths[0].parent / dest_name
        from pypdf import PdfWriter
        import tempfile
        writer = PdfWriter()
        tmp_files: list[Path] = []
        try:
            for p in paths:
                p_ext = p.suffix.lower()
                if p_ext == ".pdf":
                    writer.append(str(p))
                elif p_ext in IMAGE_EXT:
                    from PIL import Image
                    img = Image.open(p).convert("RGB")
                    tmp_p = Path(tempfile.mktemp(suffix=".pdf"))
                    tmp_files.append(tmp_p)
                    img.save(tmp_p)
                    writer.append(str(tmp_p))
                else:
                    _, text = extract(p, ocr=False)
                    if text.strip():
                        tmp_p = Path(tempfile.mktemp(suffix=".pdf"))
                        tmp_files.append(tmp_p)
                        create_pdf_from_text(tmp_p, text, title=p.stem.replace("_", " ").title())
                        writer.append(str(tmp_p))
            if len(writer.pages) == 0:
                raise ValueError("No valid pages or text found in files to merge into PDF.")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                writer.write(f)
            return dest
        finally:
            for tmp in tmp_files:
                tmp.unlink(missing_ok=True)

    # Case B: Merge into Word Doc (.docx)
    if dest_ext == ".docx":
        dest = paths[0].parent / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if all(p.suffix.lower() == ".docx" for p in paths):
            import docx
            merged = docx.Document()
            if merged.paragraphs and not merged.paragraphs[0].text:
                p_elem = merged.paragraphs[0]._element
                p_elem.getparent().remove(p_elem)
            for i, p in enumerate(paths):
                sub = docx.Document(str(p))
                if i > 0:
                    merged.add_page_break()
                for el in sub.element.body:
                    if not el.tag.endswith("sectPr"):
                        merged.element.body.append(el)
            merged.save(str(dest))
            return dest
        else:
            parts = []
            for p in paths:
                _, text = extract(p, ocr=False)
                parts.append(f"# {p.stem.replace('_', ' ').title()}\n\n{text.strip()}\n")
            return create_docx_from_markdown(dest, "\n\n---\n\n".join(parts), title=dest.stem.replace("_", " ").title())

    # Case C: Merge Images vertically
    if dest_ext in IMAGE_EXT and all(p.suffix.lower() in IMAGE_EXT for p in paths):
        dest = paths[0].parent / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in paths]
        max_w = max(img.width for img in imgs)
        total_h = sum(img.height for img in imgs)
        canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
        y = 0
        for img in imgs:
            canvas.paste(img, (0, y))
            y += img.height
        canvas.save(dest)
        return dest

    # Case D: Merge into Text / Markdown / HTML / Code
    target_ext = dest_ext or ".md"
    if not dest_name.endswith(target_ext):
        dest_name += target_ext
    dest = paths[0].parent / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for p in paths:
        _, text = extract(p, ocr=False)
        parts.append(f"# File: {p.name}\n\n{text.strip()}\n")
    full_text = "\n\n---\n\n".join(parts)

    if target_ext == ".html":
        import markdown
        html_body = markdown.markdown(full_text, extensions=["tables", "fenced_code"])
        dest.write_text(
            f"<!doctype html><html><head><meta charset='utf-8'><title>{dest.stem}</title>"
            f"<style>body{{font-family:system-ui,sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem;line-height:1.6;}}"
            f"table{{border-collapse:collapse;width:100%;margin:1rem 0;}}th,td{{border:1px solid #ddd;padding:8px;}}th{{background:#f4f4f4;}}</style>"
            f"</head><body>{html_body}</body></html>",
            encoding="utf-8",
        )
    else:
        dest.write_text(full_text, encoding="utf-8")
    return dest

