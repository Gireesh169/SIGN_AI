"""
SIGN-AI AI-Powered Vernacular Translation Engine & Cache
Multi-Backend Translation Service (Offline Canonical Dictionary + Google GTX + MyMemory + Sub-phrase Matching)
"""

import os
import sys
import json
import time
import logging
import urllib.parse
import urllib.request

logger = logging.getLogger('sign_ai.translation_engine')

SUPPORTED_LANGUAGES = {
    'hi': {'code': 'hi', 'name': 'Hindi', 'native': 'हिन्दी', 'bcp47': 'hi-IN'},
    'sat': {'code': 'sat', 'name': 'Santhali', 'native': 'ᱥᱟᱱᱛᱟᱲᱤ', 'bcp47': 'sat-IN'},
    'hoc': {'code': 'hoc', 'name': 'Ho', 'native': 'ᱦᱚ', 'bcp47': 'hoc-IN'},
    'unr': {'code': 'unr', 'name': 'Mundari', 'native': 'ᱢᱩᱱᱰᱟᱨᱤ', 'bcp47': 'unr-IN'},
    'en': {'code': 'en', 'name': 'English', 'native': 'English', 'bcp47': 'en-US'},
    'te': {'code': 'te', 'name': 'Telugu', 'native': 'తెలుగు', 'bcp47': 'te-IN'},
    'ta': {'code': 'ta', 'name': 'Tamil', 'native': 'தமிழ்', 'bcp47': 'ta-IN'},
    'kn': {'code': 'kn', 'name': 'Kannada', 'native': 'ಕನ್ನಡ', 'bcp47': 'kn-IN'},
    'ml': {'code': 'ml', 'name': 'Malayalam', 'native': 'മലയാളം', 'bcp47': 'ml-IN'},
    'bn': {'code': 'bn', 'name': 'Bengali', 'native': 'বাংলা', 'bcp47': 'bn-IN'},
    'mr': {'code': 'mr', 'name': 'Marathi', 'native': 'मराठी', 'bcp47': 'mr-IN'}
}

LANG_ALIAS_MAP = {
    'hindi': 'hi', 'hi': 'hi',
    'santhali': 'sat', 'sat': 'sat', 'santali': 'sat',
    'ho': 'hoc', 'hoc': 'hoc',
    'mundari': 'unr', 'unr': 'unr',
    'english': 'en', 'en': 'en',
    'telugu': 'te', 'te': 'te',
    'tamil': 'ta', 'ta': 'ta',
    'kannada': 'kn', 'kn': 'kn',
    'malayalam': 'ml', 'ml': 'ml',
    'bengali': 'bn', 'bn': 'bn',
    'marathi': 'mr', 'mr': 'mr'
}

# ---------------------------------------------------------------------------
# Offline Vernacular Phrase & FLN Vocabulary Dictionary
# ---------------------------------------------------------------------------
OFFLINE_DICTIONARY = {
    # --- Greetings & Common Phrases ---
    "hi": {
        "hi": "नमस्ते",
        "en": "Hi",
        "te": "హాయ్",
        "sat": "ᱡᱚᱦᱟᱨ (Johar)",
        "hoc": "ᱡᱚᱦᱟᱨ",
        "unr": "ᱡᱚᱦᱟᱨ"
    },
    "hello": {
        "hi": "नमस्ते",
        "en": "Hello",
        "te": "హలో",
        "sat": "ᱡᱚᱦᱟᱨ (Johar)",
        "hoc": "ᱡᱚᱦᱟᱨ",
        "unr": "ᱡᱚᱦᱟᱨ"
    },
    "how are you": {
        "hi": "आप कैसे हैं?",
        "sat": "ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱢᱟ?",
        "hoc": "ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱢᱟ?",
        "unr": "ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱢᱟ?",
        "te": "మీరు ఎలా ఉన్నారు? (Mīru elā unnāru?)",
        "en": "How are you?"
    },
    "hi how are you": {
        "hi": "नमस्ते, आप कैसे हैं?",
        "sat": "ᱡᱚᱦᱟᱨ, ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱢᱟ?",
        "hoc": "ᱡᱚᱦᱟᱨ, ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱢᱟ?",
        "unr": "ᱡᱚᱦᱟᱨ, ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱢᱟ?",
        "te": "హాయ్, మీరు ఎలా ఉన్నారు?",
        "en": "Hi, how are you?"
    },
    "hello how are you": {
        "hi": "नमस्ते, आप कैसे हैं?",
        "sat": "ᱡᱚᱦᱟᱨ, ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱢᱟ?",
        "hoc": "ᱡᱚᱦᱟᱨ, ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱢᱟ?",
        "unr": "ᱡᱚᱦᱟᱨ, ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱢᱟ?",
        "te": "హలో, మీరు ఎలా ఉన్నారు?",
        "en": "Hello, how are you?"
    },
    "how can i help you": {
        "hi": "मैं आपकी क्या मदद कर सकता हूँ?",
        "sat": "ᱤᱧ ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟᱧ ᱜᱚᱲᱚᱣᱟᱢᱟ?",
        "hoc": "ᱤᱧ ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟᱧ ᱜᱚᱲᱚᱣᱟᱢᱟ?",
        "unr": "ᱤᱧ ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟᱧ ᱜᱚᱲᱚᱣᱟᱢᱟ?",
        "te": "నేను మీకు ఎలా సహాయపడగలను?",
        "en": "How can I help you?"
    },
    "hi how can i help you": {
        "hi": "नमस्ते, मैं आपकी क्या मदद कर सकता हूँ?",
        "sat": "ᱡᱚᱦᱟᱨ, ᱤᱧ ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟᱧ ᱜᱚᱲᱚᱣᱟᱢᱟ?",
        "hoc": "ᱡᱚᱦᱟᱨ, ᱤᱧ ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟᱧ ᱜᱚᱲᱚᱣᱟᱢᱟ?",
        "unr": "ᱡᱚᱦᱟᱨ, ᱤᱧ ᱟᱢ ᱪᱮᱫ ᱞᱮᱠᱟᱧ ᱜᱚᱲᱚᱣᱟᱢᱟ?",
        "te": "హాయ్, నేను మీకు ఎలా సహాయపడగలను?",
        "en": "Hi, how can I help you?"
    },
    "i am fine": {
        "hi": "मैं ठीक हूँ।",
        "sat": "ᱤᱧ ᱱᱟᱯᱟᱭ ᱜᱮ ᱢᱮᱱᱟᱹᱧᱟ᱾",
        "hoc": "ᱤᱧ ᱵᱟᱹᱜᱤ ᱜᱮ ᱢᱮᱱᱟᱹᱧᱟ",
        "unr": "ᱤᱧ ᱱᱟᱯᱟᱭ ᱜᱮ ᱢᱮᱱᱟᱹᱧᱟ",
        "te": "నేను బాగున్నాను",
        "en": "I am fine."
    },
    "open your book": {
        "hi": "अपनी किताब खोलें।",
        "sat": "ᱟᱢᱟᱜ ᱯᱩᱛᱷᱤ ᱡᱷᱤᱡᱽ ᱢᱮ᱾ (Amag puthi jhij me)",
        "hoc": "ᱟᱢᱟ ᱯᱩᱛᱷᱤ ᱡᱷᱤ: ᱢᱮ (Ama puthi jhi me)",
        "unr": "ᱟᱢᱟᱜ ᱯᱩᱛᱷᱤ ᱡᱷᱤ: ᱢᱮ (Amag puthi jhi me)",
        "te": "మీ పుస్తకం తెరవండి (Mī pustakaṁ teravaṇḍi)",
        "en": "Open your book."
    },
    "please open your notebook": {
        "hi": "कृपया अपनी नोटबुक खोलें।",
        "sat": "ᱫᱟᱭᱟ ᱠᱟᱛᱮ ᱟᱢᱟᱜ ᱱᱚᱴᱵᱩᱠ ᱡᱷᱤᱡᱽ ᱢᱮ᱾",
        "hoc": "ᱫᱟᱭᱟ ᱠᱟᱛᱮ ᱱᱚᱴᱵᱩᱠ ᱡᱷᱤ: ᱢᱮ",
        "unr": "ᱫᱟᱭᱟ ᱠᱟᱛᱮ ᱱᱚᱴᱵᱩᱠ ᱡᱷᱤ: ᱢᱮ",
        "te": "దయచేసి మీ నోట్‌బుక్ తెరవండి",
        "en": "Please open your notebook."
    },
    "listen to me": {
        "hi": "मेरी बात सुनें।",
        "sat": "ᱤᱧᱟᱜ ᱠᱟᱛᱷᱟ ᱟᱸᱡᱚᱢ ᱢᱮ᱾",
        "hoc": "ᱤᱧᱟ ᱠᱟᱡᱤ ᱟᱸᱡᱚᱢ ᱢᱮ",
        "unr": "ᱤᱧᱟᱜ ᱠᱟᱡᱤ ᱟᱸᱡᱚᱢ ᱢᱮ",
        "te": "నా మాట వినండి (Nā māṭa vinaṇḍi)",
        "en": "Listen to me."
    },
    "waiting for teacher instruction...": {
        "hi": "शिक्षक के निर्देश की प्रतीक्षा कर रहे हैं...",
        "sat": "ᱢᱟᱪᱮᱛᱟᱜ ᱦᱩᱠᱩᱢ ᱞᱟᱹᱜᱤᱫ ᱛᱟᱹᱝᱜᱤ ᱨᱮ...",
        "hoc": "ᱢᱟᱪᱮᱛᱟ ᱠᱟᱡᱤ ᱞᱟᱹᱜᱤᱫ ᱛᱟᱹᱝᱜᱤ ᱨᱮ...",
        "unr": "ᱢᱟᱪᱮᱛᱟᱜ ᱠᱟᱡᱤ ᱞᱟᱹᱜᱤᱫ ᱛᱟᱹᱝᱜᱤ ᱨᱮ...",
        "te": "ఉపాధ్యాయుల సూచన కోసం వేచి చూస్తోంది...",
        "ta": "ஆசிரியரின் அறிவுறுத்தலுக்காக காத்திருக்கிறது...",
        "kn": "ಶಿಕ್ಷಕರ ಸೂಚನೆಗಾಗಿ ಕಾಯಲಾಗುತ್ತಿದೆ...",
        "ml": "അധ്യാപകന്റെ നിർദ്ദേശത്തിനായി കാത്തിരിക്കുന്നു...",
        "bn": "শিক্ষকের নির্দেশের জন্য অপেক্ষা করা হচ্ছে...",
        "mr": "शिक्षकांच्या सूचनेची वाट पाहत आहे...",
        "en": "Waiting for teacher instruction..."
    },
    "waiting for teacher instruction": {
        "hi": "शिक्षक के निर्देश की प्रतीक्षा कर रहे हैं...",
        "sat": "ᱢᱟᱪᱮᱛᱟᱜ ᱦᱩᱠᱩᱢ ᱞᱟᱹᱜᱤᱫ ᱛᱟᱹᱝᱜᱤ ᱨᱮ...",
        "hoc": "ᱢᱟᱪᱮᱛᱟ ᱠᱟᱡᱤ ᱞᱟᱹᱜᱤᱫ ᱛᱟᱹᱝᱜᱤ ᱨᱮ...",
        "unr": "ᱢᱟᱪᱮᱛᱟᱜ ᱠᱟᱡᱤ ᱞᱟᱹᱜᱤᱫ ᱛᱟᱹᱝᱜᱤ ᱨᱮ...",
        "te": "ఉపాధ్యాయుల సూచన కోసం వేచి చూస్తోంది...",
        "ta": "ஆசிரியரின் அறிவுறுத்தலுக்காக காத்திருக்கிறது...",
        "kn": "ಶಿಕ್ಷಕರ ಸೂಚನೆಗಾಗಿ ಕಾಯಲಾಗುತ್ತಿದೆ...",
        "ml": "അധ്യാപകന്റെ നിർദ്ദേശത്തിനായി കാത്തിരിക്കുന്നു...",
        "bn": "শিক্ষকের নির্দেশের জন্য অপেক্ষা করা হচ্ছে...",
        "mr": "शिक्षकांच्या सूचनेची वाट पाहत आहे...",
        "en": "Waiting for teacher instruction"
    },
    "welcome to the class": {
        "hi": "कक्षा में आपका स्वागत है।",
        "sat": "ᱟᱥᱲᱟ ᱛᱮ ᱥᱟᱹᱜᱩᱱ ᱫᱟᱨᱟᱢ",
        "hoc": "ᱟᱥᱲᱟ ᱛᱮ ᱥᱟᱹᱜᱩᱱ ᱫᱟᱨᱟᱢ",
        "unr": "ᱟᱥᱲᱟ ᱛᱮ ᱥᱟᱹᱜᱩᱱ ᱫᱟᱨᱟᱢ",
        "te": "తరగతికి స్వాగతం",
        "en": "Welcome to the class"
    },
    "what is photosynthesis?": {
        "hi": "प्रकाश संश्लेषण क्या है?",
        "sat": "ᱯᱷᱚᱴᱚᱥᱤᱱᱛᱷᱮᱥᱤᱥ ᱪᱮᱫ ᱠᱟᱱᱟ? (Photosynthesis chet kana?)",
        "hoc": "ᱯᱷᱚᱴᱚᱥᱤᱱᱛᱷᱮᱥᱤᱥ ᱪᱮᱫ ᱠᱟᱱᱟ?",
        "unr": "ᱯᱷᱚᱴᱚᱥᱤᱱᱛᱷᱮᱥᱤᱥ ᱪᱮᱫ ᱠᱟᱱᱟ?",
        "te": "కిరణజన్య సంయోగక్రియ అంటే ఏమిటి?",
        "en": "What is photosynthesis?"
    },
    "what is photosynthesis": {
        "hi": "प्रकाश संश्लेषण क्या है?",
        "sat": "ᱯᱷᱚᱴᱚᱥᱤᱱᱛᱷᱮᱥᱤᱥ ᱪᱮᱫ ᱠᱟᱱᱟ? (Photosynthesis chet kana?)",
        "hoc": "ᱯᱷᱚᱴᱚᱥᱤᱱᱛᱷᱮᱥᱤᱥ ᱪᱮᱫ ᱠᱟᱱᱟ?",
        "unr": "ᱯᱷᱚᱴᱚᱥᱤᱱᱛᱷᱮᱥᱤᱥ ᱪᱮᱫ ᱠᱟᱱᱟ?",
        "te": "కిరణజన్య సంయోగక్రియ అంటే ఏమిటి?",
        "en": "What is photosynthesis"
    },
    "please sit down": {
        "hi": "कृपया बैठ जाएं।",
        "sat": "ᱫᱟᱭᱟ ᱠᱟᱛᱮ ᱫᱩᱲᱩᱵ ᱢᱮ᱾",
        "hoc": "ᱫᱟᱭᱟ ᱠᱟᱛᱮ ᱫᱩᱵᱽ ᱢᱮ",
        "unr": "ᱫᱟᱭᱟ ᱠᱟᱛᱮ ᱫᱩᱵᱽ ᱢᱮ",
        "te": "దయచేసి కూర్చోండి (Dayacēsi kūrcōṇḍi)",
        "en": "Please sit down."
    },
    "stand up": {
        "hi": "खड़े हो जाओ।",
        "sat": "ᱵᱮᱨᱮᱫ ᱢᱮ᱾",
        "hoc": "ᱵᱮᱨᱮᱫ ᱢᱮ",
        "unr": "ᱵᱮᱨᱮᱫ ᱢᱮ",
        "te": "నిలబడండి (Nilabaḍaṇḍi)",
        "en": "Stand up."
    },
    "who can solve this": {
        "hi": "इसे कौन हल कर सकता है?",
        "sat": "ᱚᱠᱚᱭ ᱱᱤᱭᱟᱹ ᱥᱚᱞᱦᱮ ᱫᱟᱲᱮᱭᱟᱜᱼᱟ? (Okoy niya solhe dareyaga?)",
        "hoc": "ᱚᱠᱚᱭ ᱱᱤᱭᱟᱹ ᱥᱚᱞᱦᱮ ᱫᱟᱲᱮᱭᱟ?",
        "unr": "ᱚᱠᱚᱭ ᱱᱤᱭᱟᱹ ᱥᱚᱞᱦᱮ ᱫᱟᱲᱮᱭᱟ?",
        "te": "దీనిని ఎవరు పరిష్కరించగలరు?",
        "en": "Who can solve this?"
    },
    "read the sentence": {
        "hi": "वाक्य पढ़ें।",
        "sat": "ᱟᱹᱭᱟᱹᱛ ᱯᱟᱲᱦᱟᱣ ᱢᱮ᱾",
        "hoc": "ᱠᱟᱡᱤ ᱯᱟᱲᱦᱟᱣ ᱢᱮ",
        "unr": "ᱠᱟᱡᱤ ᱯᱟᱲᱦᱟᱣ ᱢᱮ",
        "te": "వాక్యాన్ని చదవండి",
        "en": "Read the sentence."
    },
    "write in your notebook": {
        "hi": "अपनी कॉपी में लिखें।",
        "sat": "ᱟᱢᱟᱜ ᱚᱞ ᱯᱩᱛᱷᱤᱨᱮ ᱚᱞ ᱢᱮ᱾",
        "hoc": "ᱟᱢᱟ ᱯᱩᱛᱷᱤᱨᱮ ᱚᱞ ᱢᱮ",
        "unr": "ᱟᱢᱟᱜ ᱯᱩᱛᱷᱤᱨᱮ ᱚᱞ ᱢᱮ",
        "te": "మీ నోట్‌బుక్‌లో రాయండి",
        "en": "Write in your notebook."
    },
    "do you understand": {
        "hi": "क्या आप समझ गए?",
        "sat": "ᱪᱮᱫ ᱟᱢᱮᱢ ᱵᱩᱡᱷᱟᱹᱣ ᱠᱮᱫᱼᱟ? (Ched amem bujhau keda?)",
        "hoc": "ᱪᱮᱫ ᱟᱢ ᱵᱩᱡᱷᱟᱹᱣ ᱠᱮᱫᱟ?",
        "unr": "ᱪᱮᱫ ᱟᱢ ᱵᱩᱡᱷᱟᱹᱣ ᱠᱮᱫᱟ?",
        "te": "మీకు అర్థమైందా? (Mīku arthamaindā?)",
        "en": "Do you understand?"
    },
    "good morning": {
        "hi": "सुप्रभात।",
        "sat": "ᱡᱚᱦᱟᱨ / ᱥᱮᱛᱟᱜ ᱡᱚᱦᱟᱨ᱾ (Johar / Setag Johar)",
        "hoc": "ᱡᱚᱦᱟᱨ (Johar)",
        "unr": "ᱡᱚᱦᱟᱨ (Johar)",
        "te": "శుభోదయం / నమస్కారం",
        "en": "Good morning."
    },
    "well done": {
        "hi": "बहुत बढ़िया!",
        "sat": "ᱟᱹᱰᱤ ᱱᱟᱯᱟᱭ! (Adi napay!)",
        "hoc": "ᱟᱹᱰᱤ ᱵᱟᱹᱜᱤ!",
        "unr": "ᱟᱹᱰᱤ ᱱᱟᱯᱟᱭ!",
        "te": "శభాష్ / చాలా బాగా చేశారు",
        "en": "Well done!"
    },
    "raise your hand": {
        "hi": "अपना हाथ उठाएं।",
        "sat": "ᱟᱢᱟᱜ ᱛᱤ ᱛᱩᱞ ᱢᱮ᱾",
        "hoc": "ᱟᱢᱟ ᱛᱤ ᱛᱩᱞ ᱢᱮ",
        "unr": "ᱟᱢᱟᱜ ᱛᱤ ᱛᱩᱞ ᱢᱮ",
        "te": "మీ చెయ్యి ఎత్తండి",
        "en": "Raise your hand."
    },
    "i need water": {
        "hi": "मुझे पानी चाहिए।",
        "sat": "ᱤᱧ ᱫᱟᱜ ᱞᱟᱹᱠᱛᱤ [ᱧᱩ] ᱤᱧᱟ᱾ (Ing dag lakti inya)",
        "hoc": "ᱤᱧ ᱫᱟ: ᱞᱟᱹᱠᱛᱤᱧᱟ",
        "unr": "ᱤᱧ ᱫᱟ: ᱞᱟᱹᱠᱛᱤᱧᱟ",
        "te": "నాకు మంచి నీళ్ళు కావాలి (Nāku manci nīḷḷu kāvāli)",
        "en": "I need water."
    },
    "i do not understand": {
        "hi": "मुझे समझ में नहीं आया।",
        "sat": "ᱤᱧ ᱵᱟᱹᱧ ᱵᱩᱡᱷᱟᱹᱣ ᱞᱮᱫᱼᱟ᱾",
        "hoc": "ᱤᱧ ᱠᱟᱧ ᱵᱩᱡᱷᱟᱹᱣ ᱠᱮᱫᱟ",
        "unr": "ᱤᱧ ᱠᱟᱧ ᱵᱩᱡᱷᱟᱹᱣ ᱠᱮᱫᱟ",
        "te": "నాకు అర్థం కాలేదు",
        "en": "I do not understand."
    },
    "help me": {
        "hi": "मेरी मदद करें।",
        "sat": "ᱤᱧ ᱜᱚᱲᱚᱣᱟᱹᱧ ᱢᱮ᱾",
        "hoc": "ᱤᱧ ᱜᱚᱲᱚᱣᱟᱹᱧ ᱢᱮ",
        "unr": "ᱤᱧ ᱜᱚᱲᱚᱣᱟᱹᱧ ᱢᱮ",
        "te": "నాకు సహాయం చేయండి",
        "en": "Help me."
    },
    "thank you": {
        "hi": "धन्यवाद।",
        "sat": "ᱥᱟᱨᱦᱟᱣ (Sarhao)",
        "hoc": "ᱥᱟᱨᱦᱟᱣ (Sarhao)",
        "unr": "ᱥᱟᱨᱦᱟᱣ (Sarhao)",
        "te": "ధన్యవాదాలు (Dhanyavādālu)",
        "en": "Thank you."
    },
    "yes i understand": {
        "hi": "हाँ, मैं समझ गया।",
        "sat": "ᱦᱚ tap ᱤᱧ ᱵᱩᱡᱷᱟᱹᱣ ᱠᱮᱫᱼᱟ᱾",
        "hoc": "ᱦᱚ tape ᱤᱧ ᱵᱩᱡᱷᱟᱹᱣ ᱠᱮᱫᱟ",
        "unr": "ᱦᱚ tape ᱤᱧ ᱵᱩᱡᱷᱟᱹᱣ ᱠᱮᱫᱟ",
        "te": "అవును, నాకు అర్థమైంది",
        "en": "Yes, I understand."
    },

    # --- Numbers (FLN Literacy) ---
    "1": {"hi": "एक (1)", "sat": "ᱢᱤᱫ (1 - Mid)", "hoc": "ᱢᱤᱫ (1)", "unr": "ᱢᱤᱫ (1)", "te": "ఒకటి (1 - Okati)", "en": "One (1)"},
    "2": {"hi": "दो (2)", "sat": "ᱵᱟᱨ (2 - Bar)", "hoc": "ᱵᱟᱨ (2)", "unr": "ᱵᱟᱨ (2)", "te": "రెండు (2 - Rendu)", "en": "Two (2)"},
    "3": {"hi": "तीन (3)", "sat": "ᱯᱮ (3 - Pe)", "hoc": "ᱯᱮ (3)", "unr": "ᱯᱮ (3)", "te": "మూడు (3 - Mūḍu)", "en": "Three (3)"},
    "4": {"hi": "चार (4)", "sat": "ᱯᱩᱱ (4 - Pun)", "hoc": "ᱯᱩᱱ (4)", "unr": "ᱯᱩᱱ (4)", "te": "నాలుగు (4 - Nālugu)", "en": "Four (4)"},
    "5": {"hi": "पाँच (5)", "sat": "ᱢᱚᱬᱮ (5 - Mone)", "hoc": "ᱢᱚᱬᱮ (5)", "unr": "ᱢᱚᱬᱮ (5)", "te": "ఐదు (5 - Aidu)", "en": "Five (5)"},
    "6": {"hi": "छह (6)", "sat": "ᱛᱩᱨᱩᱭ (6 - Turui)", "hoc": "ᱛᱩᱨᱩᱭ (6)", "unr": "ᱛᱩᱨᱩᱭ (6)", "te": "ఆరు (6 - Āru)", "en": "Six (6)"},
    "7": {"hi": "सात (7)", "sat": "ᱮᱭᱟᱭ (7 - Eyay)", "hoc": "ᱮᱭᱟᱭ (7)", "unr": "ᱮᱭᱟᱭ (7)", "te": "ఏడు (7 - Ēḍu)", "en": "Seven (7)"},
    "8": {"hi": "आठ (8)", "sat": "ᱤᱨᱟᱹᱞ (8 - Iral)", "hoc": "ᱤᱨᱟᱹᱞ (8)", "unr": "ᱤᱨᱟᱹᱞ (8)", "te": "ఎనిమిది (8 - Enimidi)", "en": "Eight (8)"},
    "9": {"hi": "नौ (9)", "sat": "ᱟᱨᱮ (9 - Are)", "hoc": "ᱟᱨᱮ (9)", "unr": "ᱟᱨᱮ (9)", "te": "తొమ్మిది (9 - Tommidi)", "en": "Nine (9)"},
    "10": {"hi": "दस (10)", "sat": "ᱜᱮᱞ (10 - Gel)", "hoc": "ᱜᱮᱞ (10)", "unr": "ᱜᱮᱞ (10)", "te": "పది (10 - Padi)", "en": "Ten (10)"},

    # --- Basic Vocabulary ---
    "water": {"hi": "पानी", "sat": "ᱫᱟᱜ (Dag)", "hoc": "ᱫᱟ:", "unr": "ᱫᱟ:", "te": "మంచి నీళ్ళు", "en": "Water"},
    "book": {"hi": "किताब", "sat": "ᱯᱩᱛᱷᱤ (Puthi)", "hoc": "ᱯᱩᱛᱷᱤ", "unr": "ᱯᱩᱛᱷᱤ", "te": "పుస్తకం", "en": "Book"},
    "school": {"hi": "विद्यालय", "sat": "ᱤᱛᱩᱱ ᱟᱥᱲᱟ (Itun Asra)", "hoc": "ᱟᱥᱲᱟ", "unr": "ᱟᱥᱲᱟ", "te": "పాఠశాల", "en": "School"},
    "teacher": {"hi": "शिक्षक", "sat": "ᱢᱟᱪᱮᱛ (Machet)", "hoc": "ᱢᱟᱪᱮᱛ", "unr": "ᱢᱟᱪᱮᱛ", "te": "ఉపాధ్యాయుడు", "en": "Teacher"},
    "student": {"hi": "छात्र", "sat": "ᱪᱮᱛᱮᱫᱤᱭᱟᱹ (Chetediya)", "hoc": "ᱪᱮᱛᱮᱫᱤᱭᱟᱹ", "unr": "ᱪᱮᱛᱮᱫᱤᱭᱟᱹ", "te": "విద్యార్థి", "en": "Student"}
}

# Translation Memory Cache for ultra-fast session lookup
_SESSION_TRANSLATION_CACHE = {}

def normalize_lang(code: str) -> str:
    """Normalizes language code/string to standard key."""
    if not code:
        return 'hi'
    clean = str(code).strip().lower()
    return LANG_ALIAS_MAP.get(clean, 'hi')

def _try_online_google_gtx(clean_text: str, src_lang: str, tgt_lang: str) -> str:
    """Fast Google Translate GTX free endpoint."""
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src_lang}&tl={tgt_lang}&dt=t&q={urllib.parse.quote(clean_text)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=1.8) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if data and isinstance(data, list) and len(data) > 0 and data[0]:
                    translated_chunks = [item[0] for item in data[0] if item and item[0]]
                    res_str = "".join(translated_chunks).strip()
                    if res_str and res_str.lower() != clean_text.lower():
                        return res_str
    except Exception as e:
        logger.debug("Google GTX translation fetch failed: %s", e)
    return None

def _try_online_mymemory(clean_text: str, src_lang: str, tgt_lang: str) -> str:
    """MyMemory API endpoint."""
    try:
        url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(clean_text)}&langpair={src_lang}|{tgt_lang}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=1.8) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                res_data = data.get('responseData', {})
                trans_text = res_data.get('translatedText', '').strip()
                if trans_text and trans_text.lower() != clean_text.lower():
                    return trans_text
    except Exception as e:
        logger.debug("MyMemory translation fetch failed: %s", e)
    return None

def _try_subphrase_dictionary(clean_text: str, src_lang: str, tgt_lang: str) -> str:
    """Attempts fuzzy token & sub-phrase dictionary replacement."""
    text_lower = clean_text.lower().strip('.,!?')
    
    # Sort keys by length descending to match longest phrases first
    sorted_keys = sorted(OFFLINE_DICTIONARY.keys(), key=len, reverse=True)
    
    working_text = text_lower
    replacements_made = 0
    
    for key in sorted_keys:
        if len(key) >= 2 and key in working_text:
            entry = OFFLINE_DICTIONARY[key]
            if tgt_lang in entry:
                working_text = working_text.replace(key, entry[tgt_lang])
                replacements_made += 1
                
    if replacements_made > 0 and working_text != text_lower:
        return working_text.capitalize()
    return None

def translate(text: str, source_language: str = 'hi', target_language: str = 'sat') -> dict:
    """
    Core Vernacular Translation Function with multi-backend fallbacks.
    """
    start_time = time.perf_counter()
    
    src_lang = normalize_lang(source_language)
    tgt_lang = normalize_lang(target_language)
    clean_text = (text or '').strip()
    
    # 1. Identity Translation (source == target or empty)
    if not clean_text or src_lang == tgt_lang:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return {
            'status': 'success',
            'source_language': src_lang,
            'target_language': tgt_lang,
            'source_text': clean_text,
            'translated_text': clean_text,
            'confidence': 1.0,
            'latency_ms': round(elapsed, 2),
            'backend_used': 'identity',
            'is_offline': True,
            'is_uncertain': False
        }

    # 2. Session Cache Look-up
    cache_key = f"{src_lang}:{tgt_lang}:{clean_text.lower()}"
    if cache_key in _SESSION_TRANSLATION_CACHE:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        cached_result = _SESSION_TRANSLATION_CACHE[cache_key].copy()
        cached_result['latency_ms'] = round(elapsed, 2)
        cached_result['backend_used'] = 'session_cache'
        return cached_result

    # 3. Offline Canonical Dictionary Exact Look-up
    text_lower = clean_text.lower().strip('.,!?')
    if text_lower in OFFLINE_DICTIONARY:
        entry = OFFLINE_DICTIONARY[text_lower]
        if tgt_lang in entry:
            translated_val = entry[tgt_lang]
            elapsed = (time.perf_counter() - start_time) * 1000.0
            result = {
                'status': 'success',
                'source_language': src_lang,
                'target_language': tgt_lang,
                'source_text': clean_text,
                'translated_text': translated_val,
                'confidence': 0.98,
                'latency_ms': round(elapsed, 2),
                'backend_used': 'offline_canonical_dictionary',
                'is_offline': True,
                'is_uncertain': False
            }
            _SESSION_TRANSLATION_CACHE[cache_key] = result
            return result

    # Also search by matching target values if translating reverse (e.g., English -> Hindi)
    for dict_key, entry in OFFLINE_DICTIONARY.items():
        if src_lang in entry and entry[src_lang].lower().strip('.,!?') == text_lower:
            if tgt_lang in entry:
                translated_val = entry[tgt_lang]
                elapsed = (time.perf_counter() - start_time) * 1000.0
                result = {
                    'status': 'success',
                    'source_language': src_lang,
                    'target_language': tgt_lang,
                    'source_text': clean_text,
                    'translated_text': translated_val,
                    'confidence': 0.95,
                    'latency_ms': round(elapsed, 2),
                    'backend_used': 'offline_canonical_dictionary',
                    'is_offline': True,
                    'is_uncertain': False
                }
                _SESSION_TRANSLATION_CACHE[cache_key] = result
                return result

    # 4. Fast Google GTX Online Translator
    gtx_trans = _try_online_google_gtx(clean_text, src_lang, tgt_lang)
    if gtx_trans:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        result = {
            'status': 'success',
            'source_language': src_lang,
            'target_language': tgt_lang,
            'source_text': clean_text,
            'translated_text': gtx_trans,
            'confidence': 0.96,
            'latency_ms': round(elapsed, 2),
            'backend_used': 'google_gtx_translator',
            'is_offline': False,
            'is_uncertain': False
        }
        _SESSION_TRANSLATION_CACHE[cache_key] = result
        return result

    # 5. MyMemory Online Translator
    mymem_trans = _try_online_mymemory(clean_text, src_lang, tgt_lang)
    if mymem_trans:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        result = {
            'status': 'success',
            'source_language': src_lang,
            'target_language': tgt_lang,
            'source_text': clean_text,
            'translated_text': mymem_trans,
            'confidence': 0.90,
            'latency_ms': round(elapsed, 2),
            'backend_used': 'mymemory_translator',
            'is_offline': False,
            'is_uncertain': False
        }
        _SESSION_TRANSLATION_CACHE[cache_key] = result
        return result

    # 6. Sub-phrase Dictionary Replacement Fallback
    subphrase_trans = _try_subphrase_dictionary(clean_text, src_lang, tgt_lang)
    if subphrase_trans:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        result = {
            'status': 'success',
            'source_language': src_lang,
            'target_language': tgt_lang,
            'source_text': clean_text,
            'translated_text': subphrase_trans,
            'confidence': 0.85,
            'latency_ms': round(elapsed, 2),
            'backend_used': 'subphrase_dictionary',
            'is_offline': True,
            'is_uncertain': False
        }
        _SESSION_TRANSLATION_CACHE[cache_key] = result
        return result

    # 7. Clean Text Fallback (Preserves Original Text without ugly error tags)
    elapsed = (time.perf_counter() - start_time) * 1000.0
    return {
        'status': 'success',
        'source_language': src_lang,
        'target_language': tgt_lang,
        'source_text': clean_text,
        'translated_text': clean_text,
        'confidence': 0.50,
        'latency_ms': round(elapsed, 2),
        'backend_used': 'original_text_pass_through',
        'is_offline': True,
        'is_uncertain': True
    }

def get_supported_languages() -> dict:
    """Returns dictionary of supported vernacular languages."""
    return SUPPORTED_LANGUAGES

if __name__ == '__main__':
    print("Testing SIGN-AI Vernacular Translation Engine...")
    t1 = translate("hi how are you", "en", "hi")
    print(f"EN -> Hindi: '{t1['source_text']}' -> '{t1['translated_text']}' (Latency: {t1['latency_ms']} ms, Backend: {t1['backend_used']})")
    
    t2 = translate("hi how are you", "en", "te")
    print(f"EN -> Telugu: '{t2['source_text']}' -> '{t2['translated_text']}' (Latency: {t2['latency_ms']} ms, Backend: {t2['backend_used']})")

    t3 = translate("hi how can i help you", "en", "hi")
    print(f"EN -> Hindi: '{t3['source_text']}' -> '{t3['translated_text']}' (Latency: {t3['latency_ms']} ms, Backend: {t3['backend_used']})")
