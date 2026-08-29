"""
SIGN-AI Vernacular Smart Classroom — Visual Flashcard Generator Module
Generates visual flashcards featuring illustrations/icons, Hindi words, target-language words,
pronunciation guides, and offline audio playback support.
"""

import os
import sys
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translation_engine import translate

logger = logging.getLogger('sign_ai.flashcard_generator')

FLASHCARD_VOCAB_PACKS = {
    "fln_numbers": [
        {"key": "1", "icon": "1️⃣", "en": "One"},
        {"key": "2", "icon": "2️⃣", "en": "Two"},
        {"key": "3", "icon": "3️⃣", "en": "Three"},
        {"key": "4", "icon": "4️⃣", "en": "Four"},
        {"key": "5", "icon": "5️⃣", "en": "Five"}
    ],
    "classroom_objects": [
        {"key": "book", "icon": "📖", "en": "Book"},
        {"key": "pen", "icon": "✏️", "en": "Pencil"},
        {"key": "water", "icon": "🚰", "en": "Water"},
        {"key": "school", "icon": "🏫", "en": "School"},
        {"key": "teacher", "icon": "👩‍🏫", "en": "Teacher"}
    ],
    "nature_environment": [
        {"key": "sun", "icon": "☀️", "en": "Sun"},
        {"key": "tree", "icon": "🌳", "en": "Tree"},
        {"key": "apple", "icon": "🍎", "en": "Apple"}
    ]
}

def generate_flashcards(pack_id: str = "classroom_objects", target_language: str = "sat") -> dict:
    """
    Generates a visual flashcard pack in Hindi + Target Vernacular Language.
    """
    items = FLASHCARD_VOCAB_PACKS.get(pack_id, FLASHCARD_VOCAB_PACKS["classroom_objects"])
    
    cards = []
    for item in items:
        en_word = item["en"]
        hi_res = translate(en_word, "en", "hi")
        tgt_res = translate(en_word, "en", target_language)
        
        cards.append({
            "card_id": f"fc_{item['key']}_{target_language}",
            "concept_key": item["key"],
            "icon": item["icon"],
            "english_word": en_word,
            "hindi_word": hi_res["translated_text"],
            "target_language": target_language,
            "target_word": tgt_res["translated_text"],
            "audio_text": tgt_res["translated_text"]
        })
        
    return {
        "status": "success",
        "pack_id": pack_id,
        "target_language": target_language,
        "total_cards": len(cards),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cards": cards
    }

def get_available_flashcard_packs() -> list:
    """Returns list of available flashcard categories."""
    return [
        {"id": "fln_numbers", "title": "FLN Numbers & Counting (संख्याएं)"},
        {"id": "classroom_objects", "title": "Classroom & Learning Objects (कक्षा वस्तुएं)"},
        {"id": "nature_environment", "title": "Nature & Environment (प्रकृति और पर्यावरण)"}
    ]

if __name__ == '__main__':
    fc = generate_flashcards("classroom_objects", "sat")
    print("Generated Santhali Flashcard Pack:", json.dumps(fc, indent=2, ensure_ascii=False))
