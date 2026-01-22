"""
XML converter for ElevenLabs multi-conversational TTS using LLM.

Uses LangChain and LLM to intelligently detect dialogue, identify speakers,
and add audio tags for ElevenLabs voice remixing and multi-voice features.
"""
import asyncio
import logging
import os
import re
import time
import uuid
from typing import Optional, Tuple
from xml.dom import minidom
from xml.etree import ElementTree as ET

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Maximum concurrent LLM requests (to stay within rate limits)
MAX_CONCURRENT_REQUESTS = 5

# Preset list of 10 character names for ElevenLabs multi-voice TTS
# These are the only speakers that will be used - all others map to "narrator"
PRESET_CHARACTERS = [
    "narrator",  # Always included
    "Frodo",
    "Sam",
    "Merry",
    "Pippin",
    "Gandalf",
    "Aragorn",
    "Legolas",
    "Gimli",
    "Gollum",
    "Tom Bombadil",
    "Boromir",
    "Galadriel",
]

def get_llm(model_name: str = "gpt-5.2", temperature: float = 0.3) -> ChatOpenAI:
    """
    Initialize LLM for dialogue and audio tag processing.
    
    Args:
        model_name: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        temperature: Temperature for generation (lower = more deterministic)
    
    Returns:
        Configured ChatOpenAI instance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with: export OPENAI_API_KEY='your-key-here'"
        )
    
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)


def safe_llm_invoke(chain, input_data: dict, max_retries: int = 3, base_delay: float = 1.0):
    """
    Safely invoke LLM chain with retry logic and error handling (synchronous version).
    
    Args:
        chain: LangChain chain to invoke
        input_data: Input data dictionary
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
    
    Returns:
        LLM response string
    
    Raises:
        Exception: If all retries fail or non-retryable error occurs
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response = chain.invoke(input_data)
            # Validate response is not empty
            if not response or not str(response).strip():
                raise ValueError("Empty response from LLM")
            return response
            
        except Exception as e:
            error_str = str(e).lower()
            error_msg = str(e)
            
            # Check for rate limit errors
            if "rate limit" in error_str or "429" in error_msg:
                wait_time = base_delay * (2 ** attempt) + (attempt * 0.5)
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.1f}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Rate limit exceeded after {max_retries} attempts: {e}")
            
            # Check for network/timeout errors
            elif any(keyword in error_str for keyword in ["connection", "timeout", "network", "unreachable"]):
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Network error after {max_retries} attempts: {e}")
            
            # Check for server errors (retryable)
            elif any(code in error_msg for code in ["500", "502", "503", "504"]):
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"Server error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Server error after {max_retries} attempts: {e}")
            
            # Check for authentication/authorization errors (non-retryable)
            elif any(keyword in error_str for keyword in ["401", "403", "authentication", "unauthorized", "forbidden", "invalid api key"]):
                raise Exception(f"Authentication error (non-retryable): {e}")
            
            # Check for invalid request errors (non-retryable)
            elif any(keyword in error_str for keyword in ["400", "bad request", "invalid"]):
                raise Exception(f"Invalid request error (non-retryable): {e}")
            
            # Check for empty/invalid response
            elif isinstance(e, ValueError) and "empty" in error_str:
                raise Exception(f"Invalid LLM response: {e}")
            
            # Unknown error - retry with exponential backoff
            else:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    last_exception = e
                    continue
                else:
                    raise Exception(f"Unexpected error after {max_retries} attempts: {e}") from last_exception
    
    raise Exception(f"Failed after {max_retries} attempts")


async def safe_llm_ainvoke(chain, input_data: dict, max_retries: int = 3, base_delay: float = 1.0):
    """
    Safely invoke LLM chain asynchronously with retry logic and error handling.
    
    Args:
        chain: LangChain chain to invoke (must support ainvoke)
        input_data: Input data dictionary
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
    
    Returns:
        LLM response string
    
    Raises:
        Exception: If all retries fail or non-retryable error occurs
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response = await chain.ainvoke(input_data)
            # Validate response is not empty
            if not response or not str(response).strip():
                raise ValueError("Empty response from LLM")
            return response
            
        except Exception as e:
            error_str = str(e).lower()
            error_msg = str(e)
            
            # Check for rate limit errors
            if "rate limit" in error_str or "429" in error_msg:
                wait_time = base_delay * (2 ** attempt) + (attempt * 0.5)
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.1f}s...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Rate limit exceeded after {max_retries} attempts: {e}")
            
            # Check for network/timeout errors
            elif any(keyword in error_str for keyword in ["connection", "timeout", "network", "unreachable"]):
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Network error after {max_retries} attempts: {e}")
            
            # Check for server errors (retryable)
            elif any(code in error_msg for code in ["500", "502", "503", "504"]):
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"Server error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Server error after {max_retries} attempts: {e}")
            
            # Check for authentication/authorization errors (non-retryable)
            elif any(keyword in error_str for keyword in ["401", "403", "authentication", "unauthorized", "forbidden", "invalid api key"]):
                raise Exception(f"Authentication error (non-retryable): {e}")
            
            # Check for invalid request errors (non-retryable)
            elif any(keyword in error_str for keyword in ["400", "bad request", "invalid"]):
                raise Exception(f"Invalid request error (non-retryable): {e}")
            
            # Check for empty/invalid response
            elif isinstance(e, ValueError) and "empty" in error_str:
                raise Exception(f"Invalid LLM response: {e}")
            
            # Unknown error - retry with exponential backoff
            else:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    last_exception = e
                    continue
                else:
                    raise Exception(f"Unexpected error after {max_retries} attempts: {e}") from last_exception
    
    raise Exception(f"Failed after {max_retries} attempts")


async def parse_dialogue_to_xml(
    text: str, 
    default_speaker: str = "narrator",
    llm: Optional[ChatOpenAI] = None
) -> str:
    """
    Convert text to XML format with speaker tags using LLM.
    
    Uses LLM to intelligently detect dialogue and identify speakers,
    handling complex patterns that regex cannot.
    
    Args:
        text: Input text to convert
        default_speaker: Speaker name for non-dialogue text (default: "narrator")
        llm: Optional LLM instance (will create one if not provided)
    
    Returns:
        XML string with speaker tags
    """
    if llm is None:
        llm = get_llm()
    
    # Create prompt for dialogue detection and speaker identification
    dialogue_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at analyzing literary text and identifying dialogue vs narration.

Your task is to convert text into XML format with speaker tags. For each segment of text, identify:
1. Whether it is dialogue (spoken by a character) or narration
2. If it's dialogue, identify the speaker name
3. If it's narration, use the narrator tag

Rules:
- Dialogue is text that characters speak (usually in quotes, but not always)
- Narration is descriptive text, thoughts, or actions
- **CRITICAL: When dialogue is embedded in narration (e.g., "'Hello!' said John." or "'Ai! ai!' wailed Legolas."), you MUST extract the dialogue to a separate speaker element. Keep only the attribution ("said John." or "wailed Legolas.") in narration. Never include dialogue quotes in narration - always extract them to the appropriate speaker.**
- Speaker names should be clean and consistent (e.g., "Frodo", "Sam", "Merry", "Pippin", "Tom Bombadil")
- If a speaker is mentioned but not clearly identified, use the most likely speaker based on context
- Handle dialogue that continues across multiple sentences or paragraphs
- Preserve the exact text content, only adding XML tags
- Do not duplicate dialogue - if dialogue appears in narration, extract it to a speaker element and remove it from narration

Output format: Return ONLY valid XML in this exact structure:
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis">
  <voice name="speaker_name">text content here</voice>
  <voice name="another_speaker">more text</voice>
</speak>

Do not include any explanation or markdown formatting, only the XML."""),
        ("user", """Convert the following text to XML with speaker tags. Identify all dialogue and assign speakers. Use "{default_speaker}" for narration.

Text to convert:
{text}""")
    ])
    
    # Create chain
    chain = dialogue_prompt | llm | StrOutputParser()
    
    # Process text in chunks if it's very long (to avoid token limits)
    max_chunk_size = 10000  # characters (≈2,500 tokens - safe for gpt-5.2)
    if len(text) <= max_chunk_size:
        try:
            xml_output = await safe_llm_ainvoke(chain, {"text": text, "default_speaker": default_speaker})
        except Exception as e:
            logger.error(f"Failed to parse dialogue to XML: {e}")
            # Fallback: return simple XML with all text as narrator
            root = ET.Element("speak")
            root.set("version", "1.0")
            root.set("xmlns", "http://www.w3.org/2001/10/synthesis")
            voice_elem = ET.SubElement(root, "voice", name=default_speaker)
            voice_elem.text = text
            return ET.tostring(root, encoding="unicode")
    else:
        # Split into paragraphs and process in batches
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        # Process chunks in parallel with concurrency limit
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async def process_chunk(chunk_text: str) -> str:
            """Process a single chunk with semaphore-controlled concurrency."""
            async with semaphore:
                try:
                    return await safe_llm_ainvoke(chain, {"text": chunk_text, "default_speaker": default_speaker})
                except Exception as e:
                    logger.error(f"Failed to process chunk: {e}")
                    # Fallback: add chunk as narrator text
                    root = ET.Element("speak")
                    root.set("version", "1.0")
                    root.set("xmlns", "http://www.w3.org/2001/10/synthesis")
                    voice_elem = ET.SubElement(root, "voice", name=default_speaker)
                    voice_elem.text = chunk_text
                    return ET.tostring(root, encoding="unicode")
        
        # Process all chunks in parallel
        xml_parts = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])
        
        # Combine XML parts (merge speak tags)
        xml_output = merge_xml_parts(xml_parts)
    
    # Clean and validate XML
    xml_output = clean_xml_output(xml_output)
    
    # Limit to preset characters (ElevenLabs API limit of 10)
    xml_output = limit_speakers_to_preset(xml_output, default_speaker=default_speaker)
    
    return prettify_xml(xml_output)


async def add_audio_tags(
    text: str, 
    llm: Optional[ChatOpenAI] = None
) -> str:
    """
    Add ElevenLabs audio tags ([laughs], [whispers], [sighs], etc.) using LLM.
    
    Uses LLM to contextually identify where audio tags should be inserted
    based on the emotional content and context of the text.
    
    Args:
        text: Input text to add tags to
        llm: Optional LLM instance (will create one if not provided)
    
    Returns:
        Text with audio tags inserted
    """
    if llm is None:
        llm = get_llm()
    
    # Create prompt for audio tag insertion
    audio_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at analyzing text for emotional and vocal cues.

Your task is to add ElevenLabs audio tags to text where appropriate. These tags come in three categories:

1. EMOTIONS AND DELIVERY TAGS (influence how the voice delivers the text):
- [laughs] - Adds laughter
- [whispers] - Converts to whispered speech
- [sighs] - Adds sighing
- [sad] - Creates a sad tone
- [giggling] - Adds a giggling effect
- [groaning] - Creates a groaning sound
- [cautiously] - Makes the voice sound cautious
- [cheerfully] - Adds a cheerful tone
- [elated] - Creates an excited, happy delivery
- [indecisive] - Makes the voice sound uncertain
- [quizzically] - Adds a questioning tone

2. AUDIO EVENT TAGS (represent environmental or contextual sounds):
- [leaves rustling] - Environmental sound
- [gentle footsteps] - Footstep sounds
- [applause] - Applause sound

3. OVERALL DIRECTION TAGS (provide broader context for delivery):
- [football] - Football context
- [wrestling match] - Wrestling context
- [auctioneer] - Auctioneer style

Rules:
- Insert tags directly into the text where they enhance the delivery
- Place tags immediately before the relevant text or phrase
- You can use multiple tags for complex emotions (e.g., "[elated] Yes! [laughs] I'm so glad!")
- Use tags to indicate interruptions with punctuation: "[cautiously] Hello, is this seat-" "[jumping in] Free?"
- Use ellipses with tags for trailing sentences: "[indecisive] Hi, can I get uhhh..."
- Look for emotional context, not just explicit words
- Preserve all original text exactly
- Tags should be in square brackets like [laughs]
- Be generous with tags - it's better to have more emotional nuance than less
- Focus primarily on Emotions and Delivery Tags, use Audio Event and Direction Tags sparingly when contextually appropriate

Output: Return the text with audio tags inserted where appropriate. Do not add any explanation or formatting."""),
        ("user", """Add ElevenLabs audio tags to the following text where appropriate:

{text}""")
    ])
    
    # Create chain
    chain = audio_prompt | llm | StrOutputParser()
    
    # Process in chunks if needed
    max_chunk_size = 10000  # characters (≈2,500 tokens - safe for gpt-5.2)
    if len(text) <= max_chunk_size:
        try:
            tagged_text = await safe_llm_ainvoke(chain, {"text": text})
        except Exception as e:
            logger.error(f"Failed to add audio tags: {e}")
            # Fallback: return original text without tags
            return text
    else:
        # Split by sentences for better context preservation
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append('. '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        # Process chunks in parallel with concurrency limit
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async def process_chunk(chunk_text: str) -> str:
            """Process a single chunk with semaphore-controlled concurrency."""
            async with semaphore:
                try:
                    return await safe_llm_ainvoke(chain, {"text": chunk_text})
                except Exception as e:
                    logger.error(f"Failed to add audio tags to chunk: {e}")
                    # Fallback: use original chunk without tags
                    return chunk_text
        
        # Process all chunks in parallel
        tagged_parts = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])
        
        tagged_text = ' '.join(tagged_parts)
    
    return tagged_text


async def add_audio_tags_to_dialogue_only(
    xml_string: str,
    default_speaker: str = "narrator",
    llm: Optional[ChatOpenAI] = None
) -> str:
    """
    Add audio tags only to dialogue elements, not narration.
    
    Batches all dialogue segments together to minimize API calls.
    
    Args:
        xml_string: XML string with voice elements
        default_speaker: Speaker name that represents narration (won't get tags)
        llm: Optional LLM instance (will create one if not provided)
    
    Returns:
        XML string with audio tags added only to dialogue
    """
    if llm is None:
        llm = get_llm()
    
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return xml_string
    
    # Find all voice elements, handling namespaces
    voice_elements = []
    for elem in root.iter():
        tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag_name == "voice":
            voice_elements.append(elem)
    
    # Collect dialogue elements (non-narrator) with their indices
    dialogue_elements = []
    dialogue_texts = []
    
    for voice_elem in voice_elements:
        speaker_name = voice_elem.get("name", default_speaker)
        
        # Skip narration
        if speaker_name == default_speaker:
            continue
        
        # Get the dialogue text
        dialogue_text = voice_elem.text or ""
        if not dialogue_text.strip():
            continue
        
        dialogue_elements.append(voice_elem)
        dialogue_texts.append(dialogue_text)
    
    # If no dialogue found, return as-is
    if not dialogue_texts:
        return ET.tostring(root, encoding="unicode")
    
    # Use a unique delimiter that's extremely unlikely to appear in text
    # Using a UUID-like pattern with explicit markers
    delimiter = f"__DIALOGUE_SEGMENT_{uuid.uuid4().hex[:16]}__"
    
    # Verify delimiter doesn't appear in any dialogue text (extremely unlikely)
    for text in dialogue_texts:
        if delimiter in text:
            # If delimiter appears in text, use a different one
            delimiter = f"__DIALOGUE_SEGMENT_{uuid.uuid4().hex[:16]}_ALT__"
            break
    
    # Combine all dialogue segments with delimiter
    combined_dialogue = delimiter.join(dialogue_texts)
    
    # Process the combined dialogue with audio tags
    tagged_combined = await add_audio_tags(combined_dialogue, llm=llm)
    
    # Split back into individual segments
    tagged_segments = tagged_combined.split(delimiter)
    
    # Handle case where delimiter might have been modified by LLM
    # If split didn't work (wrong number of segments), try to recover
    if len(tagged_segments) != len(dialogue_elements):
        # Fallback: try to find delimiter variations or process individually
        # This shouldn't happen often, but handle gracefully
        import re

        # Try to find the delimiter or similar patterns
        if delimiter not in tagged_combined:
            # LLM might have removed/modified delimiter, process individually as fallback
            for i, voice_elem in enumerate(dialogue_elements):
                original_text = dialogue_texts[i]
                tagged_dialogue = await add_audio_tags(original_text, llm=llm)
                voice_elem.text = tagged_dialogue
        else:
            # Split worked but count mismatch - use what we have
            for i, voice_elem in enumerate(dialogue_elements):
                if i < len(tagged_segments):
                    voice_elem.text = tagged_segments[i].strip()
    else:
        # Successfully split - map back to elements
        for i, voice_elem in enumerate(dialogue_elements):
            voice_elem.text = tagged_segments[i].strip()
    
    return ET.tostring(root, encoding="unicode")


def extract_text_from_xml(xml_string: str) -> str:
    """
    Extract all text content from XML, removing tags and audio tags.
    
    Args:
        xml_string: XML string with voice elements
        
    Returns:
        Plain text extracted from XML
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        # Fallback: use regex to extract text
        # Remove XML tags and audio tags
        text = re.sub(r'<[^>]+>', '', xml_string)
        text = re.sub(r'\[[^\]]+\]', '', text)  # Remove audio tags like [laughs]
        return text.strip()
    
    # Extract all text from voice elements
    texts = []
    for elem in root.iter():
        tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag_name == "voice":
            text = elem.text or ""
            # Remove audio tags from text
            text = re.sub(r'\[[^\]]+\]', '', text)
            if text.strip():
                texts.append(text.strip())
    
    return " ".join(texts)


def normalize_text_for_comparison(text: str) -> str:
    """
    Normalize text for comparison (remove extra whitespace, normalize quotes, etc.).
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    # Remove audio tags if present
    text = re.sub(r'\[[^\]]+\]', '', text)
    return text.strip().lower()


def validate_xml_content(original_text: str, xml_string: str, 
                          tolerance: float = 0.05) -> Tuple[bool, dict]:
    """
    Validate that XML contains all original text content.
    
    Args:
        original_text: Original text before conversion
        xml_string: Generated XML string
        tolerance: Acceptable loss percentage (default: 5%)
        
    Returns:
        Tuple of (is_valid, stats_dict)
        stats_dict contains: original_length, xml_length, missing_chars, 
                            missing_percentage, missing_sentences
    """
    # Extract text from XML
    xml_text = extract_text_from_xml(xml_string)
    
    # Normalize for comparison
    original_norm = normalize_text_for_comparison(original_text)
    xml_norm = normalize_text_for_comparison(xml_text)
    
    # Calculate lengths
    original_len = len(original_norm)
    xml_len = len(xml_norm)
    missing_chars = original_len - xml_len
    missing_percentage = (missing_chars / original_len * 100) if original_len > 0 else 0
    
    # Check for missing sentences (split by periods)
    original_sentences = set(s.strip() for s in original_text.split('.') if s.strip())
    xml_sentences = set(s.strip() for s in xml_text.split('.') if s.strip())
    missing_sentences = original_sentences - xml_sentences
    
    # Check for missing paragraphs
    original_paras = [p.strip() for p in original_text.split('\n\n') if p.strip()]
    xml_paras_text = [p.strip() for p in xml_text.split('\n\n') if p.strip()]
    missing_paras = []
    for para in original_paras:
        para_norm = normalize_text_for_comparison(para)
        found = False
        for xml_para in xml_paras_text:
            if para_norm in normalize_text_for_comparison(xml_para):
                found = True
                break
        if not found:
            missing_paras.append(para[:100] + "..." if len(para) > 100 else para)
    
    stats = {
        "original_length": original_len,
        "xml_length": xml_len,
        "missing_chars": missing_chars,
        "missing_percentage": missing_percentage,
        "missing_sentences_count": len(missing_sentences),
        "missing_paragraphs_count": len(missing_paras),
        "missing_paragraphs": missing_paras[:5],  # Limit to first 5
    }
    
    # Consider valid if missing percentage is within tolerance
    is_valid = missing_percentage <= (tolerance * 100)
    
    return is_valid, stats


def merge_consecutive_speakers(xml_string: str) -> str:
    """
    Merge consecutive voice elements with the same speaker name.
    
    This fixes cases where the LLM splits narration into multiple segments
    even though the speaker doesn't change.
    
    Example:
    <voice name="narrator">text1</voice>
    <voice name="narrator">text2</voice>
    <voice name="Gandalf">dialogue</voice>
    <voice name="narrator">text3</voice>
    
    Becomes:
    <voice name="narrator">text1 text2</voice>
    <voice name="Gandalf">dialogue</voice>
    <voice name="narrator">text3</voice>
    
    Args:
        xml_string: XML string with voice elements
        
    Returns:
        XML string with consecutive speakers merged
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        # If parsing fails, return as-is
        logger.warning("Failed to parse XML for merging consecutive speakers, returning as-is")
        return xml_string
    
    # Find all voice elements, handling namespaces
    voice_elements = []
    for elem in root.iter():
        tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag_name == "voice":
            voice_elements.append(elem)
    
    if not voice_elements:
        return xml_string
    
    # Get namespace from root if present
    namespace = None
    if "}" in root.tag:
        namespace = root.tag.split("}")[0] + "}"
    
    # Create new root with same namespace and attributes
    new_root = ET.Element(root.tag)
    for key, value in root.attrib.items():
        new_root.set(key, value)
    
    # Merge consecutive elements with same speaker
    current_speaker = None
    current_text_parts = []
    
    for voice_elem in voice_elements:
        speaker_name = voice_elem.get("name", "narrator")
        text_content = (voice_elem.text or "").strip()
        
        # Skip empty text segments
        if not text_content:
            continue
        
        # If same speaker as previous, accumulate text
        if speaker_name == current_speaker and current_speaker is not None:
            current_text_parts.append(text_content)
        else:
            # Different speaker (or first element) - save previous and start new
            if current_speaker is not None and current_text_parts:
                # Create merged voice element with proper namespace
                merged_text = " ".join(current_text_parts)
                if namespace:
                    voice_tag = f"{namespace}voice"
                else:
                    voice_tag = "voice"
                merged_elem = ET.SubElement(new_root, voice_tag, name=current_speaker)
                merged_elem.text = merged_text
            
            # Start new speaker
            current_speaker = speaker_name
            current_text_parts = [text_content]
    
    # Don't forget the last speaker
    if current_speaker is not None and current_text_parts:
        merged_text = " ".join(current_text_parts)
        if namespace:
            voice_tag = f"{namespace}voice"
        else:
            voice_tag = "voice"
        merged_elem = ET.SubElement(new_root, voice_tag, name=current_speaker)
        merged_elem.text = merged_text
    
    return ET.tostring(new_root, encoding="unicode")


async def convert_text_to_elevenlabs_xml(
    text: str, 
    default_speaker: str = "narrator",
    add_audio_tags_flag: bool = True,
    llm: Optional[ChatOpenAI] = None,
    validate_content: bool = True,
    content_tolerance: float = 0.05
) -> str:
    """
    Complete conversion: text -> XML with speakers -> audio tags on dialogue only.
    
    Args:
        text: Input text
        default_speaker: Default speaker name for narration
        add_audio_tags_flag: Whether to add audio tags (only to dialogue, not narration)
        llm: Optional LLM instance (will create one if not provided)
        validate_content: Whether to validate that XML contains all original content
        content_tolerance: Acceptable content loss percentage (default: 5%)
    
    Returns:
        XML string ready for ElevenLabs TTS
    
    Raises:
        ValueError: If content validation fails and too much content is missing (>20%)
    """
    if llm is None:
        llm = get_llm()
    
    # Step 1: Convert to XML with speaker tags first
    xml_output = await parse_dialogue_to_xml(text, default_speaker=default_speaker, llm=llm)
    
    # Step 2: Merge consecutive narration segments (post-processing fix)
    xml_output = merge_consecutive_speakers(xml_output)
    
    # Step 3: Add audio tags only to dialogue (not narration)
    if add_audio_tags_flag:
        xml_output = await add_audio_tags_to_dialogue_only(xml_output, default_speaker=default_speaker, llm=llm)
    
    # Step 4: Validate content if requested
    if validate_content:
        is_valid, stats = validate_xml_content(text, xml_output, tolerance=content_tolerance)
        
        if not is_valid:
            logger.warning(
                f"Content validation failed: {stats['missing_percentage']:.2f}% content missing. "
                f"Missing {stats['missing_sentences_count']} sentences, "
                f"{stats['missing_paragraphs_count']} paragraphs."
            )
            
            if stats['missing_paragraphs']:
                logger.warning(f"Sample missing paragraphs: {stats['missing_paragraphs']}")
            
            # Raise exception if too much is missing
            if stats['missing_percentage'] > 20:  # More than 20% missing
                raise ValueError(
                    f"Too much content missing from XML ({stats['missing_percentage']:.2f}%). "
                    f"Original: {stats['original_length']} chars, XML: {stats['xml_length']} chars. "
                    f"Missing {stats['missing_sentences_count']} sentences, "
                    f"{stats['missing_paragraphs_count']} paragraphs."
                )
    
    return xml_output


def convert_text_to_elevenlabs_xml_sync(
    text: str, 
    default_speaker: str = "narrator",
    add_audio_tags_flag: bool = True,
    llm: Optional[ChatOpenAI] = None,
    validate_content: bool = True,
    content_tolerance: float = 0.05
) -> str:
    """
    Synchronous wrapper for convert_text_to_elevenlabs_xml.
    
    This function allows the async version to be called from synchronous code.
    
    Args:
        text: Input text
        default_speaker: Default speaker name for narration
        add_audio_tags_flag: Whether to add audio tags (only to dialogue, not narration)
        llm: Optional LLM instance (will create one if not provided)
        validate_content: Whether to validate that XML contains all original content
        content_tolerance: Acceptable content loss percentage (default: 5%)
    
    Returns:
        XML string ready for ElevenLabs TTS
    
    Raises:
        ValueError: If content validation fails and too much content is missing (>20%)
    """
    return asyncio.run(convert_text_to_elevenlabs_xml(
        text, default_speaker, add_audio_tags_flag, llm, validate_content, content_tolerance
    ))


def merge_xml_parts(xml_parts: list) -> str:
    """Merge multiple XML parts into a single speak element."""
    root = ET.Element("speak")
    root.set("version", "1.0")
    root.set("xmlns", "http://www.w3.org/2001/10/synthesis")
    
    for xml_part in xml_parts:
        try:
            part_root = ET.fromstring(xml_part)
            # Extract all voice elements from this part
            for voice_elem in part_root.findall(".//{http://www.w3.org/2001/10/synthesis}voice"):
                # Remove namespace for simplicity
                voice_name = voice_elem.get("name", "narrator")
                voice_text = voice_elem.text or ""
                if voice_text.strip():
                    new_voice = ET.SubElement(root, "voice", name=voice_name)
                    new_voice.text = voice_text
        except ET.ParseError:
            # If XML parsing fails, try to extract voice elements with regex fallback
            import re
            voice_matches = re.findall(r'<voice\s+name="([^"]+)"[^>]*>([^<]+)</voice>', xml_part)
            for voice_name, voice_text in voice_matches:
                if voice_text.strip():
                    new_voice = ET.SubElement(root, "voice", name=voice_name)
                    new_voice.text = voice_text.strip()
    
    return ET.tostring(root, encoding="unicode")


def clean_xml_output(xml_string: str) -> str:
    """Clean and validate XML output from LLM."""
    # Remove markdown code blocks if present
    xml_string = xml_string.strip()
    if xml_string.startswith("```"):
        # Remove markdown code block markers
        lines = xml_string.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        xml_string = "\n".join(lines)
    
    # Try to parse and re-serialize to ensure valid XML
    try:
        root = ET.fromstring(xml_string)
        return ET.tostring(root, encoding="unicode")
    except ET.ParseError:
        # If parsing fails, try to extract just the speak element
        import re
        speak_match = re.search(r'<speak[^>]*>.*?</speak>', xml_string, re.DOTALL)
        if speak_match:
            return speak_match.group(0)
        # Last resort: return as-is
        return xml_string


def limit_speakers_to_preset(
    xml_string: str, 
    default_speaker: str = "narrator",
    preset_characters: Optional[list] = None
) -> str:
    """
    Limit speakers to a preset list of characters (max 10 for ElevenLabs API).
    
    Maps detected speakers to the closest match in the preset list using fuzzy matching.
    If no close match is found, maps to default_speaker (usually "narrator").
    
    Args:
        xml_string: XML string with voice elements
        default_speaker: Speaker name to use for unmatched speakers
        preset_characters: List of preset character names (defaults to PRESET_CHARACTERS)
    
    Returns:
        XML string with speakers limited to preset list
    """
    if preset_characters is None:
        preset_characters = PRESET_CHARACTERS
    
    # Ensure default_speaker is in the list
    if default_speaker not in preset_characters:
        preset_characters = [default_speaker] + [c for c in preset_characters if c != default_speaker]
    
    # Normalize preset characters (lowercase for matching)
    preset_lower = {name.lower(): name for name in preset_characters}
    
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        # If parsing fails, return as-is
        return xml_string
    
    # Find all voice elements, handling namespaces
    voice_elements = []
    for elem in root.iter():
        tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag_name == "voice":
            voice_elements.append(elem)
    
    # Map each speaker to preset list
    for voice_elem in voice_elements:
        speaker_name = voice_elem.get("name", default_speaker)
        
        # If already in preset list, keep it
        if speaker_name in preset_characters:
            continue
        
        # Try exact case-insensitive match
        speaker_lower = speaker_name.lower()
        if speaker_lower in preset_lower:
            voice_elem.set("name", preset_lower[speaker_lower])
            continue
        
        # Try fuzzy matching (check if preset name is contained in speaker name or vice versa)
        matched = False
        for preset_lower_name, preset_name in preset_lower.items():
            if preset_lower_name in speaker_lower or speaker_lower in preset_lower_name:
                voice_elem.set("name", preset_name)
                matched = True
                break
        
        # If no match found, use default_speaker
        if not matched:
            voice_elem.set("name", default_speaker)
    
    return ET.tostring(root, encoding="unicode")


def prettify_xml(xml_string: str) -> str:
    """Format XML string with proper indentation."""
    try:
        dom = minidom.parseString(xml_string)
        return dom.toprettyxml(indent="  ")
    except Exception:
        # Fallback to basic formatting
        return xml_string
        
