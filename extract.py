import sys
from pathlib import Path

try:
    # Prefer PyPDF2 (widely available) but fall back to pypdf if needed
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(
            "No PDF reader library found. Please install one of: PyPDF2 or pypdf",
            file=sys.stderr,
        )
        raise exc


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in getattr(reader, "pages", []):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        texts.append(text)
    return "\n\n".join(texts).strip()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    pdf_path = project_dir / "wellness.pdf"
    out_path = project_dir / "wellness.txt"

    if not pdf_path.exists():
        print(f"PDF not found at: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    text = extract_text_from_pdf(pdf_path)

    if not text:
        print("Warning: Extracted text is empty. The PDF might be scanned or image-based.", file=sys.stderr)

    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote extracted text to: {out_path}")


if __name__ == "__main__":
    main()


