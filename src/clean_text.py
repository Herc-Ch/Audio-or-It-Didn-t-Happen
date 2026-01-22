import re
import unicodedata
from pathlib import Path


def strip_control_chars(s: str) -> str:
    # Remove Unicode control chars (category "Cc"), but keep newlines and tabs
    out = []
    for ch in s:
        if ch in ("\n", "\t"):
            out.append(ch)
        elif unicodedata.category(ch) == "Cc":
            continue
        else:
            out.append(ch)
    return "".join(out)


def fix_private_use_area_chars(text: str) -> str:
    """
    Fix Unicode Private Use Area characters that should be lowercase letters or numbers.
    
    Some PDFs use custom fonts that map lowercase letters and numbers to Private Use Area
    characters (U+F000-U+FFFF). This function maps them back to standard Unicode.
    """
    # Build a comprehensive mapping by checking common ranges
    replacements = {}
    
    # Map lowercase letters (a-z) from Private Use Area
    # Common ranges that map to a-z: U+F761-U+F77A, U+F6E1-U+F6FA, U+F7A1-U+F7BA
    for i, letter in enumerate('abcdefghijklmnopqrstuvwxyz'):
        # Range 1: U+F761-U+F77A (most common)
        replacements[chr(0xF761 + i)] = letter
        # Range 2: U+F6E1-U+F6FA (alternative)
        old_char = chr(0xF6E1 + i)
        if old_char not in replacements:
            replacements[old_char] = letter
        # Range 3: U+F7A1-U+F7BA (another alternative)
        old_char = chr(0xF7A1 + i)
        if old_char not in replacements:
            replacements[old_char] = letter
    
    # Map numbers (0-9) from Private Use Area
    # Common ranges: U+F730-U+F739, U+F6D0-U+F6D9, U+F790-U+F799, U+F770-U+F779
    for i, digit in enumerate('0123456789'):
        # Range 1: U+F730-U+F739
        replacements[chr(0xF730 + i)] = digit
        # Range 2: U+F6D0-U+F6D9
        old_char = chr(0xF6D0 + i)
        if old_char not in replacements:
            replacements[old_char] = digit
        # Range 3: U+F790-U+F799
        old_char = chr(0xF790 + i)
        if old_char not in replacements:
            replacements[old_char] = digit
        # Range 4: U+F770-U+F779 (common for numbers)
        old_char = chr(0xF770 + i)
        if old_char not in replacements:
            replacements[old_char] = digit
    
    # Apply replacements
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    
    return text


def remove_running_headers_and_pages(text: str) -> str:
    lines = text.splitlines()
    out = []

    for line in lines:
        s = line.strip()
        if not s:
            out.append("")  # keep blank lines
            continue

        low = s.lower()

        # Drop standalone page numbers: "31"
        if re.fullmatch(r"\d{1,4}", s):
            continue

        # Drop "chapter title + page number" headers like: "a long-expected party 31"
        # (allow various dashes in title)
        if re.fullmatch(r"[a-z][a-z \-–—']{3,}\s+\d{1,4}", low):
            continue

        # Drop running header lines like: "the fellowship of the ring"
        if low in {
            "the fellowship of the ring",
            "the two towers",
            "the return of the king", 
        }:
            continue

        # Drop divider lines like "***" or "* * *"
        if re.fullmatch(r"(\*\s*){3,}", s):
            continue

        out.append(line)

    return "\n".join(out)


def clean_text(text: str) -> str:
    # 0) Fix Private Use Area characters first (before other processing)
    text = fix_private_use_area_chars(text)
    
    # 1) Remove control chars like U+0002 and friends
    text = strip_control_chars(text)

    # 2) Remove running headers / page numbers / dividers
    text = remove_running_headers_and_pages(text)

    # 3) Join hyphenated line breaks: "tram-\npling" -> "trampling"
    text = re.sub(r"(\w)[\-‐–]\n(\w)", r"\1\2", text)

    # 4) Remove soft hyphen (U+00AD)
    text = text.replace("\u00ad", "")

    # 5) Remove replacement char if present
    text = text.replace("", "")

    # 6) Merge single newlines inside paragraphs (keep blank lines)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # join "word\n\nword" when it's clearly a line-wrap, not a real paragraph
    text = re.sub(r"(?<=\w)\n\n(?=\w)", " ", text)

    # 7) Normalize extra whitespace/newlines
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove inline page header prefixes like "32 the fellowship of the ring "
    text = re.sub(
        r"\b\d{1,4}\s+the fellowship of the ring\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{1,4}\s+the two towers\s+", "", text, flags=re.I)
    text = re.sub(r"\b\d{1,4}\s+the return of the king\s+", "", text, flags=re.I)

    return text.strip()


def clean_text_file(path: Path) -> None:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    path.write_text(clean_text(raw), encoding="utf-8")


def clean_dir(dir_path: Path, pattern: str = "*.txt") -> None:
    for p in dir_path.glob(pattern):
        clean_text_file(p)



