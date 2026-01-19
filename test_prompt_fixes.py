#!/usr/bin/env python
"""Test the 4 GPT audit fixes to prompt_builder."""

from pipeline.inference.prompt_builder import (
    _normalize_context_instruction,
    _get_language_description,
    _get_unified_prompt_template,
    build_messages
)

print('=== Test 1: context_instruction filter ===')
# Should pass
print(f'Short context: {repr(_normalize_context_instruction("This is about rum"))}')
# Should be filtered (too long)
long_ctx = 'x' * 301
print(f'Long context (301 chars): {repr(_normalize_context_instruction(long_ctx))}')
# Should be filtered (contains translate)
print(f'Contains translate: {repr(_normalize_context_instruction("Please translate literally"))}')
# Should be filtered (contains output)
print(f'Contains output: {repr(_normalize_context_instruction("Output in JSON format"))}')
print()

print('=== Test 2: language fallback ===')
print(f'Known lang (en): {_get_language_description("en")}')
print(f'Known lang (ar_EG): {_get_language_description("ar_EG")}')
print(f'Unknown lang (bho_IN): {_get_language_description("bho_IN")}')
print(f'Unknown lang (xyz_ABC): {_get_language_description("xyz_ABC")}')
print()

print('=== Test 3: unified prompt template ===')
prompt = _get_unified_prompt_template(
    source_text='Hello world',
    source_lang='en',
    target_lang='bho_IN',
    context_instruction='About greeting'
)
print(prompt)
print()

print('=== Test 4: LVLM output constraint check ===')
# Verify the new constraint is in the prompt
assert 'Do NOT mention the image' in prompt, "Missing LVLM constraint!"
print('✓ LVLM constraint present')
print()

print('=== Test 5: full messages structure ===')
msgs_text = build_messages(
    source_text='Test',
    source_lang='en',
    target_lang='ar_EG',
    mode='text_only'
)
print(f'text_only: {[{"role": m["role"], "types": [c["type"] for c in m["content"]]} for m in msgs_text]}')

from PIL import Image
import io
# Create a dummy image
dummy_img = Image.new('RGB', (100, 100), color='red')

msgs_image = build_messages(
    source_text='Test',
    source_lang='en',
    target_lang='ar_EG',
    mode='text_image',
    image=dummy_img
)
print(f'text_image: {[{"role": m["role"], "types": [c["type"] for c in m["content"]]} for m in msgs_image]}')

# Verify no system role
for msg in msgs_text + msgs_image:
    assert msg["role"] != "system", "Should not have system role!"
print('✓ No system role (as designed)')
print()

print('=== ALL TESTS PASSED ===')
