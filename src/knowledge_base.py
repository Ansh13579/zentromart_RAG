"""Load, parse and chunk the ZentroMart Markdown knowledge base."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_KB_DIR = (
    Path(__file__).resolve().parents[2]
    / "Capstone_Project_Dataset_FTTT_2026"
    / "kb"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "indexes" / "chunks.json"
)

SECTION_PATTERN = re.compile(
    r"^##\s+\[([^\]]+)\]\s*(.*?)\s*$",
    re.MULTILINE,
)
FRONTMATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    doc_id: str
    section_id: str
    title: str
    doc_type: str
    status: str
    effective_from: str
    text: str


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}, content

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, content[match.end() :]


def _document_title(metadata: dict[str, str], body: str, path: Path) -> str:
    if metadata.get("title"):
        return metadata["title"]

    heading = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
    return heading.group(1).strip() if heading else path.stem


def _sections(body: str, fallback_title: str) -> list[tuple[str, str, str]]:
    matches = list(SECTION_PATTERN.finditer(body))
    if not matches:
        text = body.strip()
        return [("DOCUMENT", fallback_title, text)] if text else []

    sections: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        if section_text:
            sections.append(
                (match.group(1).strip(), match.group(2).strip(), section_text)
            )
    return sections


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        current: list[str] = []
        length = 0
        index = start
        while index < len(words):
            word = words[index]
            additional = len(word) if not current else len(word) + 1
            if current and length + additional > chunk_size:
                break
            current.append(word)
            length += additional
            index += 1

        chunks.append(" ".join(current))
        if index >= len(words):
            break

        overlap_length = 0
        next_start = index
        while next_start > start and overlap_length < overlap:
            next_start -= 1
            overlap_length += len(words[next_start]) + 1
        start = next_start

    return chunks


def load_knowledge_base(
    kb_dir: Path,
    *,
    chunk_size: int = 1800,
    overlap: int = 250,
) -> list[KnowledgeChunk]:
    """Parse every Markdown document in ``kb_dir`` into metadata-rich chunks."""

    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge-base directory was not found: {kb_dir}")

    chunks: list[KnowledgeChunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        metadata, body = _parse_frontmatter(content)
        title = _document_title(metadata, body, path)
        doc_id = metadata.get("doc_id", path.stem)

        for section_id, section_title, section_text in _sections(body, title):
            parts = _split_text(section_text, chunk_size, overlap)
            for part_number, text in enumerate(parts, start=1):
                chunk_id = f"{doc_id}:{section_id}:{part_number}"
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        section_id=section_id,
                        title=section_title or title,
                        doc_type=metadata.get("doc_type", ""),
                        status=metadata.get("status", ""),
                        effective_from=metadata.get("effective_from", ""),
                        text=text,
                    )
                )
    return chunks


def save_chunks(chunks: list[KnowledgeChunk], output_path: Path) -> None:
    """Save chunks as UTF-8 JSON for the retrieval phase."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse and chunk the ZentroMart knowledge base."
    )
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=250)
    args = parser.parse_args()

    chunks = load_knowledge_base(
        args.kb_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    save_chunks(chunks, args.output)
    print(f"Created {len(chunks)} chunks at {args.output}")


if __name__ == "__main__":
    main()
