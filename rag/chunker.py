"""
Document loading and chunking for email documents.

Strategy: Each email is parsed into structured metadata (subject, sender,
receiver) plus body text. The body is split using a sliding-window character
splitter. Most emails are ~750 chars and fit in a single chunk at size 500,
preserving full context. Longer emails get 2-3 chunks with overlap to avoid
losing information at boundaries.
"""

import os
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _parse_email(filepath: str) -> dict:
    """Parse a raw email .txt file into structured fields."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    subject = ""
    sender_name, sender_email = "", ""
    receiver_name, receiver_email = "", ""
    body = ""

    lines = content.strip().split("\n")
    body_start = 0

    for i, line in enumerate(lines):
        if line.startswith("Subject:"):
            subject = line[len("Subject:"):].strip()
        elif line.startswith("From:"):
            match = re.match(r"From:\s*(.+?)\s*<(.+?)>", line)
            if match:
                sender_name, sender_email = match.group(1).strip(), match.group(2).strip()
        elif line.startswith("To:"):
            match = re.match(r"To:\s*(.+?)\s*<(.+?)>", line)
            if match:
                receiver_name, receiver_email = match.group(1).strip(), match.group(2).strip()
        elif line.strip() == "" and i > 0 and body_start == 0:
            prev_lines = "\n".join(lines[:i])
            if "To:" in prev_lines:
                body_start = i + 1

    body = "\n".join(lines[body_start:]).strip()

    return {
        "subject": subject,
        "sender_name": sender_name,
        "sender_email": sender_email,
        "receiver_name": receiver_name,
        "receiver_email": receiver_email,
        "body": body,
        "source": os.path.basename(filepath),
    }


def load_emails(directory: str = "emails") -> list[dict]:
    """Load all .txt email files from a directory."""
    emails = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            emails.append(_parse_email(filepath))
    return emails


def _split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Split text into chunks using a character-based sliding window.
    Attempts to break at sentence boundaries when possible.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to break at a sentence boundary (. ! ?) within the last 20% of the chunk
            search_start = start + int(chunk_size * 0.8)
            best_break = -1
            for punct in [".\n", ". ", "! ", "? "]:
                pos = text.rfind(punct, search_start, end)
                if pos > best_break:
                    best_break = pos + len(punct)

            if best_break > search_start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap
        if start >= len(text):
            break

    return chunks


def chunk_documents(
    emails: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Convert parsed emails into chunks with metadata.

    Each chunk's text is prefixed with email metadata (subject, sender, receiver)
    so the embedding captures the full context even for body-only chunks.
    """
    all_chunks = []

    for email in emails:
        header = (
            f"Subject: {email['subject']}\n"
            f"From: {email['sender_name']} <{email['sender_email']}>\n"
            f"To: {email['receiver_name']} <{email['receiver_email']}>\n\n"
        )

        body_chunks = _split_text(email["body"], chunk_size, chunk_overlap)

        for i, body_text in enumerate(body_chunks):
            chunk_text = header + body_text

            all_chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "subject": email["subject"],
                    "sender_name": email["sender_name"],
                    "sender_email": email["sender_email"],
                    "receiver_name": email["receiver_name"],
                    "receiver_email": email["receiver_email"],
                    "source": email["source"],
                    "chunk_index": i,
                    "total_chunks": len(body_chunks),
                },
            ))

    return all_chunks
