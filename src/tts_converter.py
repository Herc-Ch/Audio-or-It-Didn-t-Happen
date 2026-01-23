"""
TTS converter for ElevenLabs multi-voice audiobook generation.
Converts XML or TXT files to MP3 using ElevenLabs API.
Supports request stitching for better prosody continuity in text pipeline.
"""
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment

load_dotenv()

# Voice mapping for characters
VOICE_DICT = {
    "narrator": "1t8J0GKJDtHkrTD6Vwym",
    "Frodo": "MObdX0C5tJP7lQGGLAso",
    "Pippin": "ppu4Fa4iALOcBl7pFxJg",
    "Sam": "sbLlkinBhIPBlbnrUqqB",
    "Merry": "ppu4Fa4iALOcBl7pFxJg",
    "Gandalf": "Bs9Ov5tNnDfd87RgGQ9h",
    "Gollum": "U2nrqYPuipkbWKORt1eP",
    "Legolas": "ZPk226mSYGb01kYZxlT0",
    "Aragorn": "LQ0UMyjs058i24znw9dp",
    "Gimli": "xhm1Br6laXAmAMKHPMuS",
    "Boromir": "R12FBaW0oGzYkypQap1h",
    "Tom Bombadil": "sB9BeRajhZYfj5OXOCrM",
    "Galadriel": "fAr1gTHp351NqVNEFQN2",
}


def merge_short_narrator_segments(segments, max_narrator_length=50):
    """
    Merge short narrator segments with adjacent character segments
    to avoid jarring voice switches for brief interjections like "said Frodo;"
    """
    if not segments:
        return segments
    
    merged = []
    i = 0
    
    while i < len(segments):
        character, text = segments[i]
        
        # If it's a short narrator segment
        if character == "narrator" and len(text) <= max_narrator_length:
            # Priority: if next segment is same character as previous, merge with next
            if (i + 1 < len(segments) and merged and 
                merged[-1][0] == segments[i+1][0] != "narrator"):
                # Merge narrator into next segment (keeps character voice consistent)
                next_char, next_text = segments[i+1]
                merged.append((next_char, text + " " + next_text))
                i += 2
                continue
            # Otherwise, merge with previous character segment
            elif merged and merged[-1][0] != "narrator":
                # Merge with previous character segment
                merged[-1] = (merged[-1][0], merged[-1][1] + " " + text)
                i += 1
                continue
        
        # Keep segment as-is
        merged.append((character, text))
        i += 1
    
    return merged


def parse_xml(xml_file_path: Path):
    """Parse XML file and extract voice segments."""
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    ns = {'ns0': 'http://www.w3.org/2001/10/synthesis'}
    
    segments = []
    for voice_elem in root.findall('.//ns0:voice', ns):
        character = voice_elem.get('name', 'narrator')
        text = voice_elem.text or ""
        if text.strip():
            segments.append((character, text))
    
    # Merge short narrator segments to avoid jarring voice switches
    segments = merge_short_narrator_segments(segments, max_narrator_length=50)
    
    return segments


def parse_txt(txt_file_path: Path):
    """Parse TXT file - all text uses narrator voice."""
    content = txt_file_path.read_text(encoding='utf-8')
    
    # Split by paragraphs (double newlines) for better processing
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    # Return all as narrator segments
    segments = [("narrator", para) for para in paragraphs]
    return segments


def convert_to_mp3(
    file_path: Path,
    output_file: Path,
    max_workers: int = 5,
    use_request_stitching: bool = False
) -> bool:
    """
    Convert XML or TXT file to MP3 audiobook using ElevenLabs TTS.
    
    Args:
        file_path: Path to input XML or TXT file
        output_file: Path to output MP3 file
        max_workers: Maximum concurrent TTS requests (only used if not using request stitching)
        use_request_stitching: Use request stitching for better prosody continuity
                              (only works with eleven_multilingual_v2 model, requires sequential processing)
        
    Returns:
        True if successful, False otherwise
    """
    # Check API key
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("⚠️  Error: ELEVENLABS_API_KEY not set in environment")
        return False
    
    elevenlabs = ElevenLabs(api_key=api_key)
    
    # Detect file type and parse accordingly
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.xml':
        segments = parse_xml(file_path)
    elif file_ext == '.txt':
        segments = parse_txt(file_path)
    else:
        print(f"⚠️  Error: Unsupported file type: {file_ext}. Only .xml and .txt are supported.")
        return False
    
    if not segments:
        print("⚠️  Warning: No segments found in file.")
        return False
    
    # Determine model based on file type
    # Request stitching only works with eleven_multilingual_v2
    if use_request_stitching:
        model_id = "eleven_multilingual_v2"
        print("🔗 Using request stitching for better prosody continuity...")
    else:
        model_id = "eleven_multilingual_v2" if file_ext == '.txt' else "eleven_v3"
    
    # Use request stitching for text pipeline (TXT files)
    if use_request_stitching:
        return _convert_with_request_stitching(
            elevenlabs, segments, output_file, model_id
        )
    
    # Standard parallel processing for XML files
    def generate_segment(i, character, text):
        """Generate audio for a single segment."""
        voice_id = VOICE_DICT.get(character, VOICE_DICT["narrator"])
        
        try:
            audio = elevenlabs.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format="mp3_44100_128",
            )
        except Exception as e:
            # If voice_id is invalid, fallback to narrator
            print(f"⚠️  Warning: Invalid voice_id for {character}, using narrator instead. Error: {e}")
            voice_id = VOICE_DICT["narrator"]
            audio = elevenlabs.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format="mp3_44100_128",
            )
        
        # Save audio to file
        temp_file = f"temp_{i:04d}.mp3"
        with open(temp_file, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        
        # Validate file was written correctly
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
            raise ValueError(f"Failed to write audio file {temp_file}")
        
        # Try to validate MP3 by attempting to load it
        try:
            test_audio = AudioSegment.from_mp3(temp_file)
            if len(test_audio) == 0:
                raise ValueError(f"Audio file {temp_file} is empty")
        except Exception as e:
            raise ValueError(f"Invalid MP3 file {temp_file}: {e}")
        
        return i, temp_file
    
    # Process segments in parallel
    print(f"🎙️  Generating audio for {len(segments)} segments...")
    audio_files_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(generate_segment, i, character, text): i 
            for i, (character, text) in enumerate(segments)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                i, temp_file = future.result()
                audio_files_dict[i] = temp_file
                print(f"  ✓ Completed segment {i+1}/{len(segments)}")
            except Exception as e:
                print(f"  ⚠️  Error processing segment: {e}")
    
    # Sort audio files by index to maintain order
    audio_files = [audio_files_dict[i] for i in sorted(audio_files_dict.keys())]
    
    # Combine audio segments
    print(f"🔗 Combining {len(audio_files)} audio segments...")
    combined = AudioSegment.empty()
    skipped_files = []
    for file in audio_files:
        try:
            if os.path.exists(file) and os.path.getsize(file) > 0:
                audio = AudioSegment.from_mp3(file)
                combined += audio
                combined += AudioSegment.silent(duration=300)  # 300ms pause between segments
            else:
                print(f"⚠️  Warning: Skipping invalid file {file} (missing or empty)")
                skipped_files.append(file)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load {file}, skipping: {e}")
            skipped_files.append(file)
    
    if skipped_files:
        print(f"⚠️  Warning: {len(skipped_files)} file(s) were skipped due to errors")
    
    if len(combined) > 0:
        combined.export(str(output_file), format="mp3")
        print(f"✅ Successfully created {output_file}")
    else:
        print("❌ Error: No valid audio segments to combine")
        return False
    
    # Clean up temp files
    for file in audio_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                print(f"⚠️  Warning: Could not remove {file}: {e}")
    
    return True


def _convert_with_request_stitching(
    elevenlabs: ElevenLabs,
    segments: list,
    output_file: Path,
    model_id: str
) -> bool:
    """
    Convert segments to MP3 using request stitching for better prosody continuity.
    
    Request stitching requires sequential processing and only works with
    eleven_multilingual_v2 model. It maintains voice prosody across segments.
    
    Args:
        elevenlabs: ElevenLabs client instance
        segments: List of (character, text) tuples
        output_file: Path to output MP3 file
        model_id: Model ID (must be eleven_multilingual_v2)
        
    Returns:
        True if successful, False otherwise
    """
    if model_id != "eleven_multilingual_v2":
        print("⚠️  Warning: Request stitching only works with eleven_multilingual_v2. Falling back to standard processing.")
        return False
    
    print(f"🎙️  Generating audio for {len(segments)} segments with request stitching...")
    
    request_ids = []
    audio_buffers = []
    
    # Process segments sequentially (required for request stitching)
    for i, (character, text) in enumerate(segments):
        voice_id = VOICE_DICT.get(character, VOICE_DICT["narrator"])
        
        try:
            # Use with_raw_response to get request ID from headers
            with elevenlabs.text_to_speech.with_raw_response.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                previous_request_ids=request_ids,
                output_format="mp3_44100_128",
            ) as response:
                # Extract request ID from response headers
                request_id = response._response.headers.get("request-id")
                if request_id:
                    request_ids.append(request_id)
                
                # Read all audio data from the stream
                audio_data = b''.join(chunk for chunk in response.data)
                audio_buffers.append(BytesIO(audio_data))
                
                print(f"  ✓ Completed segment {i+1}/{len(segments)}")
                
        except Exception as e:
            print(f"  ⚠️  Error processing segment {i+1}: {e}")
            # Fallback to narrator voice if character voice fails
            if character != "narrator":
                try:
                    voice_id = VOICE_DICT["narrator"]
                    with elevenlabs.text_to_speech.with_raw_response.convert(
                        text=text,
                        voice_id=voice_id,
                        model_id=model_id,
                        previous_request_ids=request_ids,
                        output_format="mp3_44100_128",
                    ) as response:
                        request_id = response._response.headers.get("request-id")
                        if request_id:
                            request_ids.append(request_id)
                        audio_data = b''.join(chunk for chunk in response.data)
                        audio_buffers.append(BytesIO(audio_data))
                        print(f"  ✓ Completed segment {i+1}/{len(segments)} (using narrator fallback)")
                except Exception as e2:
                    print(f"  ❌ Failed to process segment {i+1} even with fallback: {e2}")
                    return False
            else:
                return False
    
    # Combine all audio buffers
    print(f"🔗 Combining {len(audio_buffers)} audio segments...")
    combined_audio_data = b''.join(buffer.getvalue() for buffer in audio_buffers)
    
    # Save to temporary file first for validation
    temp_file = "temp_stitched.mp3"
    with open(temp_file, "wb") as f:
        f.write(combined_audio_data)
    
    # Validate the combined audio
    try:
        test_audio = AudioSegment.from_mp3(temp_file)
        if len(test_audio) == 0:
            print("❌ Error: Combined audio file is empty")
            os.remove(temp_file)
            return False
    except Exception as e:
        print(f"❌ Error: Invalid combined MP3 file: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False
    
    # Export final file
    test_audio.export(str(output_file), format="mp3")
    print(f"✅ Successfully created {output_file}")
    
    # Clean up temp file
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"⚠️  Warning: Could not remove {temp_file}: {e}")
    
    return True
