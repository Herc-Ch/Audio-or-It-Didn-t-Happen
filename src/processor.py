import subprocess
from pathlib import Path

from glossapi import Corpus

from src.clean_text import clean_dir
from src.xml_converter import convert_text_to_elevenlabs_xml_sync, validate_xml_content


def md_to_txt(md_path: Path, txt_path: Path):
    """Convert markdown to plain text using pandoc."""
    subprocess.run(
        ["pandoc", str(md_path), "-t", "plain", "-o", str(txt_path)],
        check=True,
    )


def convert_txt_to_xml(txt_dir: Path, xml_dir: Path, default_speaker: str = "narrator"):
    """
    Convert text files to ElevenLabs XML format with speaker tags and audio annotations.
    
    Args:
        txt_dir: Directory containing text files
        xml_dir: Output directory for XML files
        default_speaker: Default speaker name for narration
    """
    xml_dir.mkdir(exist_ok=True)
    
    for txt_file in txt_dir.glob("*.txt"):
        xml_file = xml_dir / (txt_file.stem + ".xml")
        text_content = txt_file.read_text(encoding="utf-8", errors="ignore")
        
        print(f"Processing {txt_file.name}...")
        
        # Convert to XML with audio tags (uses LLM for dialogue detection and tagging)
        # Uses async processing internally with parallel chunk processing
        xml_content = convert_text_to_elevenlabs_xml_sync(
            text_content,
            default_speaker=default_speaker,
            add_audio_tags_flag=True,  # Always True when converting to XML
            validate_content=True,  # Enable content validation
            content_tolerance=0.05  # Allow 5% content loss
        )
        
        xml_file.write_text(xml_content, encoding="utf-8")
        
        # Log validation results
        is_valid, stats = validate_xml_content(text_content, xml_content)
        if not is_valid:
            print(f"  ⚠️  Warning: {stats['missing_percentage']:.2f}% content missing "
                  f"({stats['missing_sentences_count']} sentences, "
                  f"{stats['missing_paragraphs_count']} paragraphs)")
        else:
            print(f"  ✓ Validation passed: {stats['xml_length']} chars in XML")


def process_pdfs(input_dir: Path, output_dir: Path, convert_to_xml: bool = True):
    """
    Main processing pipeline using glossapi.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Output directory for processed files
        convert_to_xml: Whether to convert text files to ElevenLabs XML format
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    corpus = Corpus(input_dir, output_dir)
    
    # Phase 1: Extract PDFs
    corpus.extract(
        input_format="pdf",
        # accel_type="CPU",
    )
    
    markdown_dir = output_dir / "markdown"
    assert markdown_dir.exists(), "Markdown folder not found"
    
    # Clean markdown
    clean_dir(markdown_dir, pattern="*.md")
    
    # Convert to txt (for TTS)
    txt_dir = output_dir / "tts_txt"
    txt_dir.mkdir(exist_ok=True)
    
    for md_file in markdown_dir.glob("*.md"):
        txt_file = txt_dir / (md_file.stem + ".txt")
        md_to_txt(md_file, txt_file)
    
    # Convert to XML with speaker tags and audio annotations (optional)
    if convert_to_xml:
        xml_dir = output_dir / "elevenlabs_xml"
        convert_txt_to_xml(txt_dir, xml_dir)
    
    print(f"✓ Processing complete! Output: {output_dir}")
    print(f"  - Text files: {txt_dir}")
    if convert_to_xml:
        print(f"  - XML files: {output_dir / 'elevenlabs_xml'}")