"""
Prompt building utilities for translation inference.

CRITICAL: Returns multimodal message format, NOT strings.
Qwen2-VL requires image as a separate modality block in messages.
"""

from typing import Optional, Dict, Any, List, Union
from PIL import Image


# =============================================================================
# UNIFIED PROMPT TEMPLATE
# =============================================================================
# 
# CRITICAL DESIGN PRINCIPLE:
# - text-only and text-image prompts MUST be semantically identical
# - The ONLY difference is whether an image block is included
# - This ensures fair A/B comparison
#

def _normalize_context_instruction(ctx: Optional[str]) -> Optional[str]:
    """
    Filter context_instruction to prevent prompt pollution.
    
    Context should be a "weak hint", not a second prompt.
    """
    if ctx is None:
        return None
    
    # Prevent long instructions from dominating the prompt
    if len(ctx) > 300:
        return None
    
    # Prevent task override instructions
    ctx_lower = ctx.lower()
    forbidden_patterns = [
        "translate",
        "summarize", 
        "paraphrase",
        "rewrite",
        "output",
        "respond",
    ]
    for pattern in forbidden_patterns:
        if pattern in ctx_lower:
            return None
    
    return ctx


def _get_language_description(lang_code: str) -> str:
    """
    Get a robust language description for the prompt.
    
    For unknown language codes, use a fallback that LVLMs understand better
    than raw codes like 'bho_IN'.
    """
    lang_names = get_language_names()
    name = lang_names.get(lang_code)
    if name:
        return name
    # Fallback: wrap in descriptive text for better LVLM understanding
    return f"the language '{lang_code}'"


def _get_unified_prompt_template(
    source_text: str,
    source_lang: str,
    target_lang: str,
    context_instruction: Optional[str] = None
) -> str:
    """
    Get the unified prompt template used for BOTH text-only and text-image.
    
    This ensures the only experimental variable is the presence of the image.
    """
    src_desc = _get_language_description(source_lang)
    tgt_desc = _get_language_description(target_lang)
    
    # Base template with strong output constraints (including LVLM-specific)
    template = f"""Translate the following text from {src_desc} to {tgt_desc}.
Use any provided context only if it helps improve translation accuracy.
Output ONLY the translation in {tgt_desc}.
Do NOT include explanations, descriptions, or the original text.
Do NOT mention the image, visual content, or context explicitly.
"""
    
    # Add filtered context instruction as supplementary info (NOT as replacement)
    filtered_ctx = _normalize_context_instruction(context_instruction)
    if filtered_ctx:
        template += f"""\nAdditional context:
{filtered_ctx}
"""
    
    template += f"""\nSource text:
{source_text}

Translation:"""
    
    return template


# =============================================================================
# MAIN API: build_messages
# =============================================================================

def build_messages(
    source_text: str,
    source_lang: str,
    target_lang: str,
    mode: str,
    image: Optional[Image.Image] = None,
    context_instruction: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Build chat messages for Qwen2-VL inference.
    
    CRITICAL: Returns proper multimodal message format, not strings.
    
    Args:
        source_text: Text to translate
        source_lang: Source language code  
        target_lang: Target language code
        mode: "text_only" or "text_image"
        image: PIL Image object (required if mode="text_image")
        context_instruction: Optional additional context
        
    Returns:
        List of message dicts in Qwen2-VL format
    """
    # Get unified prompt text (identical for both modes)
    prompt_text = _get_unified_prompt_template(
        source_text=source_text,
        source_lang=source_lang,
        target_lang=target_lang,
        context_instruction=context_instruction
    )
    
    if mode == "text_only":
        # Text-only: just text content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
    elif mode == "text_image":
        # Text-image: image block + identical text
        if image is None:
            raise ValueError("image is required for text_image mode")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'text_only' or 'text_image'")
    
    return messages


# =============================================================================
# DEPRECATED: Keep for backward compatibility but log warning
# =============================================================================

def build_translation_prompt(
    source_text: str,
    source_lang: str,
    target_lang: str,
    context_instruction: Optional[str] = None
) -> str:
    """
    DEPRECATED: Use build_messages() instead.
    
    This returns a string which breaks multimodal protocol.
    Kept only for backward compatibility.
    """
    import warnings
    warnings.warn(
        "build_translation_prompt() is deprecated. Use build_messages() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _get_unified_prompt_template(
        source_text=source_text,
        source_lang=source_lang,
        target_lang=target_lang,
        context_instruction=context_instruction
    )


def build_multimodal_prompt(
    source_text: str,
    source_lang: str,
    target_lang: str,
    context_instruction: Optional[str] = None
) -> str:
    """
    DEPRECATED: Use build_messages() instead.
    
    This returns a string which breaks multimodal protocol.
    Kept only for backward compatibility.
    """
    import warnings
    warnings.warn(
        "build_multimodal_prompt() is deprecated. Use build_messages() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _get_unified_prompt_template(
        source_text=source_text,
        source_lang=source_lang,
        target_lang=target_lang,
        context_instruction=context_instruction
    )


def get_language_names() -> Dict[str, str]:
    """Return mapping of language codes to full names."""
    return {
        # ISO 639-1 codes
        "en": "English",
        "zh": "Chinese",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "cs": "Czech",
        "pt": "Portuguese",
        "it": "Italian",
        "nl": "Dutch",
        "pl": "Polish",
        "ar": "Arabic",
        "hi": "Hindi",
        "th": "Thai",
        "vi": "Vietnamese",
        "tr": "Turkish",
        "uk": "Ukrainian",
        # Locale-specific codes
        "zh_CN": "Simplified Chinese",
        "zh_TW": "Traditional Chinese",
        "de_DE": "German",
        "en_US": "English",
        "en_GB": "English",
        "pt_BR": "Portuguese",
        "es_ES": "Spanish",
        "es_MX": "Spanish",
        "ar_EG": "Arabic",
        "ar_SA": "Arabic",
        "fr_FR": "French",
        "ja_JP": "Japanese",
        "ko_KR": "Korean",
        "ru_RU": "Russian",
    }


# NOTE: get_system_prompt() was intentionally removed.
# Reason: We use ONLY user role for consistency between text-only and text-image.
# Adding system prompt would introduce another variable in A/B comparison.
# If needed in future, it MUST be added identically to BOTH modes.
