"""
Main pipeline for PDF to Speech conversion.
Processes PDFs through extraction, cleaning, XML conversion, and optional TTS generation.
"""
import argparse
from pathlib import Path

from src.processor import process_pdfs
from src.tts_converter import convert_to_mp3


def main():
    parser = argparse.ArgumentParser(
        description="PDF to Speech Pipeline - Convert PDFs to XML and optionally to MP3 audiobooks"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="pdfs",
        help="Directory containing PDF files (default: pdfs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--no-xml",
        action="store_true",
        help="Skip XML conversion (only generate cleaned text)"
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Generate MP3 audiobook from XML files after conversion"
    )
    parser.add_argument(
        "--tts-format",
        choices=["xml", "txt"],
        default="xml",
        help="Format to use for TTS conversion: xml (multi-voice) or txt (narrator only) (default: xml)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist")
        return
    
    print("🚀 Starting PDF to Speech pipeline...")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print()
    
    # Step 1: Process PDFs
    print("📄 Step 1: Processing PDFs...")
    process_pdfs(
        input_dir=input_dir,
        output_dir=output_dir,
        convert_to_xml=not args.no_xml
    )
    print()
    
    # Step 2: Generate TTS (if requested)
    if args.tts:
        print("🎙️  Step 2: Generating audiobook...")
        
        if args.tts_format == "xml":
            xml_dir = output_dir / "elevenlabs_xml"
            if not xml_dir.exists():
                print(f"❌ Error: XML directory '{xml_dir}' not found. Run without --no-xml first.")
                return
            
            xml_files = list(xml_dir.glob("*.xml"))
            if not xml_files:
                print(f"⚠️  No XML files found in {xml_dir}")
                return
            
            for xml_file in xml_files:
                output_mp3 = output_dir / "audiobooks" / f"{xml_file.stem}.mp3"
                output_mp3.parent.mkdir(exist_ok=True)
                
                print(f"   Processing: {xml_file.name}")
                success = convert_to_mp3(xml_file, output_mp3)
                if success:
                    print(f"   ✅ Created: {output_mp3}")
                else:
                    print(f"   ❌ Failed to create: {output_mp3}")
                print()
        
        else:  # txt format
            txt_dir = output_dir / "tts_txt"
            if not txt_dir.exists():
                print(f"❌ Error: TXT directory '{txt_dir}' not found.")
                return
            
            txt_files = list(txt_dir.glob("*.txt"))
            if not txt_files:
                print(f"⚠️  No TXT files found in {txt_dir}")
                return
            
            for txt_file in txt_files:
                output_mp3 = output_dir / "audiobooks" / f"{txt_file.stem}.mp3"
                output_mp3.parent.mkdir(exist_ok=True)
                
                print(f"   Processing: {txt_file.name}")
                # Use request stitching for text pipeline (better prosody)
                success = convert_to_mp3(txt_file, output_mp3, use_request_stitching=True)
                if success:
                    print(f"   ✅ Created: {output_mp3}")
                else:
                    print(f"   ❌ Failed to create: {output_mp3}")
                print()
    
    print("✅ Pipeline complete!")


if __name__ == "__main__":
    main()
