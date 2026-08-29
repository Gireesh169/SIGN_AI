"""
SIGN-AI Multilingual Translator Module
Provides post-sign recognition translation from canonical English to target Indian languages.
Features:
- Configurable supported languages dictionary
- Clean interface: translate(text, target_language) -> dict
- Session translation caching using (source_text, target_language) tuple key
- Robust multi-stage translation: Session Cache -> Built-in Phrase Dictionary -> Online HTTP Translator -> English Fallback
- Render cloud deployment compatible without heavy local model dependencies
"""

import os
import sys
import logging
import threading
import urllib.request
import urllib.parse
import json

logger = logging.getLogger('sign_ai.translator')

# ---------------------------------------------------------------------------
# Supported Languages Configuration
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'native': 'English', 'bcp47': 'en-US'},
    'te': {'name': 'Telugu', 'native': 'తెలుగు', 'bcp47': 'te-IN'},
    'hi': {'name': 'Hindi', 'native': 'हिन्दी', 'bcp47': 'hi-IN'},
    'ta': {'name': 'Tamil', 'native': 'தமிழ்', 'bcp47': 'ta-IN'},
    'kn': {'name': 'Kannada', 'native': 'கன்னட', 'bcp47': 'kn-IN'},
    'ml': {'name': 'Malayalam', 'native': 'മലയാളം', 'bcp47': 'ml-IN'},
    'bn': {'name': 'Bengali', 'native': 'বাংলা', 'bcp47': 'bn-IN'},
    'mr': {'name': 'Marathi', 'native': 'मराठी', 'bcp47': 'mr-IN'}
}

# Mapping of alternative names / codes to standardized language code
LANGUAGE_CODE_MAP = {
    'english': 'en', 'en': 'en',
    'telugu': 'te', 'te': 'te',
    'hindi': 'hi', 'hi': 'hi',
    'tamil': 'ta', 'ta': 'ta',
    'kannada': 'kn', 'kn': 'kn',
    'malayalam': 'ml', 'ml': 'ml',
    'bengali': 'bn', 'bn': 'bn',
    'marathi': 'mr', 'mr': 'mr'
}

# ---------------------------------------------------------------------------
# Pre-compiled Canonical Offline Phrase Dictionary for High-Speed Zero-Latency
# ---------------------------------------------------------------------------
CANONICAL_PHRASE_DICTIONARY = {
    # Common Sign Dataset Sentences & Classroom Instructions
    "i need water": {
        "te": "నాకు నీళ్లు కావాలి",
        "hi": "मुझे पानी चाहिए",
        "ta": "எனக்கு தண்ணீர் வேண்டும்",
        "kn": "ನನಗೆ ನೀರು ಬೇಕು",
        "ml": "എനിക്ക് വെള്ളം വേണം",
        "bn": "আমার জল দরকার",
        "mr": "मला पाणी हवे आहे"
    },
    "please open your notebook": {
        "te": "దయచేసి మీ నోట్బుక్ తెరవండి",
        "hi": "कृपया अपनी नोटबुक खोलें",
        "ta": "தயவுசெய்து உங்கள் நோட்டுப் புத்தகத்தைத் திறக்கவும்",
        "kn": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ನೋಟ್‌ಬುಕ್ ತೆರೆಯಿರಿ",
        "ml": "ദയവായി നിങ്ങളുടെ നോട്ട്ബുക്ക് തുറക്കൂ",
        "bn": "দয়া করে আপনার নোটবুকটি খুলুন",
        "mr": "कृपया तुमची वही उघडा"
    },
    "how are you": {
        "te": "మీరు ఎలా ఉన్నారు?",
        "hi": "आप कैसे हैं?",
        "ta": "நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "kn": "ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "ml": "സുഖമാണോ?",
        "bn": "আপনি কেমন আছেন?",
        "mr": "तुम्ही कसे आहात?"
    },
    "hi how are you": {
        "te": "హాయ్ మీరు ఎలా ఉన్నారు?",
        "hi": "नमस्ते आप कैसे हैं?",
        "ta": "ஹாய் நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "kn": "ಹಾಯ್ ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "ml": "ഹായ് സുഖമാണോ?",
        "bn": "হাই আপনি কেমন আছেন?",
        "mr": "हाय तुम्ही कसे आहात?"
    },
    "help me": {
        "te": "నాకు సహాయం చేయండి",
        "hi": "मेरी मदद करो",
        "ta": "எனக்கு உதவுங்கள்",
        "kn": "ನನಗೆ ಸಹಾಯ ಮಾಡಿ",
        "ml": "എന്നെ സഹായിക്കൂ",
        "bn": "আমাকে সাহায্য করুন",
        "mr": "मला मदत करा"
    },
    "i need help": {
        "te": "నాకు సహాయం కావాలి",
        "hi": "मुझे मदद की ज़रूरत है",
        "ta": "எனக்கு உதவி தேவை",
        "kn": "ನನಗೆ ಸಹಾಯ ಬೇಕು",
        "ml": "എനിക്ക് സഹായം വേണം",
        "bn": "আমার সাহায্য দরকার",
        "mr": "मला मदतीची गरज आहे"
    },
    "thank you so much": {
        "te": "చాలా ధన్యవాదాలు",
        "hi": "बहुत बहुत धन्यवाद",
        "ta": "மிக்க நன்றி",
        "kn": "ತುಂಬಾ ಧನ್ಯವಾದಗಳು",
        "ml": "വളരെ നന്ദി",
        "bn": "আপনাকে অনেক ধন্যবাদ",
        "mr": "खूप खूप धन्यवाद"
    },
    "you are welcome": {
        "te": "మీకు స్వాగతం",
        "hi": "आपका स्वागत है",
        "ta": "உங்களை வரவேற்கிறோம்",
        "kn": "ನಿಮಗೆ ಸ್ವಾಗತ",
        "ml": "തീർച്ചയായും",
        "bn": "আপনাকে স্বাগতম",
        "mr": "तुमचे स्वागत आहे"
    },
    "bring water for me": {
        "te": "నా కోసం నీరు తీసుకురండి",
        "hi": "मेरे लिए पानी लाओ",
        "ta": "எனக்கு தண்ணீர் கொண்டு வாருங்கள்",
        "kn": "ನನಗೆ ನೀರು ತನ್ನಿ",
        "ml": "എനിക്ക് വെള്ളം കൊണ്ടുവരൂ",
        "bn": "আমার জন্য জল আনুন",
        "mr": "माझ्यासाठी पाणी आणा"
    },
    "can i help you": {
        "te": "నేను మీకు సహాయం చేయవచ్చా?",
        "hi": "क्या मैं आपकी मदद कर सकता हूँ?",
        "ta": "நான் உங்களுக்கு உதவலாமா?",
        "kn": "ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಬಹುದೇ?",
        "ml": "ഞാൻ നിങ്ങളെ സഹായിക്കട്ടെ?",
        "bn": "আমি কি আপনাকে সাহায্য করতে পারি?",
        "mr": "मी तुम्हाला मदत करू शका का?"
    },
    "congratulations": {
        "te": "అభినందనలు",
        "hi": "बधाई हो",
        "ta": "வாழ்த்துக்கள்",
        "kn": "ಅಭಿನಂದನೆಗಳು",
        "ml": "അഭിനന്ദനങ്ങൾ",
        "bn": "অভিনন্দন",
        "mr": "अभिनंदन"
    },
    "i am hungry": {
        "te": "నాకు ఆకలిగా ఉంది",
        "hi": "मुझे भूख लगी है",
        "ta": "எனக்கு பசிக்கிறது",
        "kn": "ನನಗೆ ಹಸಿವಾಗಿದೆ",
        "ml": "എനിക്ക് വിശക്കുന്നു",
        "bn": "আমার ক্ষুধা পেয়েছে",
        "mr": "मला भूक लागली आहे"
    },
    "i am fine thank you sir": {
        "te": "నేను బాగున్నాను ధన్యవాదాలు సార్",
        "hi": "मैं ठीक हूँ धन्यवाद सर",
        "ta": "நான் நலமாக இருக்கிறேன் நன்றி ஐயா",
        "kn": "ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ ಧನ್ಯವಾದಗಳು ಸರ್",
        "ml": "എനിക്ക് സുഖമാണ് നന്ദി സർ",
        "bn": "আমি ভালো আছি ধন্যবাদ স্যার",
        "mr": "मी ठीक आहे धन्यवाद सर"
    },
    "nice to meet you": {
        "te": "మిమ్మల్ని కలిసినందుకు సంతోషంగా ఉంది",
        "hi": "आपसे मिलकर अच्छा लगा",
        "ta": "உங்களை சந்தித்ததில் மகிழ்ச்சி",
        "kn": "ನಿಮ್ಮನ್ನು ಭೇಟಿಯಾಗಿದ್ದು ಸಂತೋಷವಾಯಿತು",
        "ml": "കണ്ടുമുട്ടിയതിൽ സന്തോഷം",
        "bn": "আপনার সাথে দেখা করে ভালো লাগলো",
        "mr": "तुम्हाला भेटून आनंद झाला"
    },
    "what are you doing": {
        "te": "మీరు ఏమి చేస్తున్నారు?",
        "hi": "आप क्या कर रहे हैं?",
        "ta": "நீங்கள் என்ன செய்கிறீர்கள்?",
        "kn": "ನೀವು ಏನು ಮಾಡುತ್ತಿದ್ದೀರಿ?",
        "ml": "നിങ്ങൾ എന്താണ് ചെയ്യുന്നത്?",
        "bn": "আপনি কি করছেন?",
        "mr": "तुम्ही काय करत आहात?"
    },
    "what is your name": {
        "te": "మీ పేరు ఏమిటి?",
        "hi": "आपका नाम क्या है?",
        "ta": "உங்கள் பெயர் என்ன?",
        "kn": "ನಿಮ್ಮ ಹೆಸರೇನು?",
        "ml": "നിങ്ങളുടെ പേരെന്താണ്?",
        "bn": "আপনার নাম কি?",
        "mr": "तुमचे नाव काय आहे?"
    }
}


class MultilingualTranslator:
    """
    Multilingual Translation Engine for SIGN-AI.
    Translates canonical English text output into selected target languages.
    Uses multi-tiered caching & fallback mechanism.
    """
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        logger.info("MultilingualTranslator initialized with %d supported languages", len(SUPPORTED_LANGUAGES))

    def _normalize_lang(self, lang_input: str) -> str:
        """Converts language string/code to standard 2-letter code."""
        if not lang_input:
            return 'en'
        clean = str(lang_input).strip().lower()
        return LANGUAGE_CODE_MAP.get(clean, clean)

    def _online_google_translate(self, text: str, target_lang: str) -> str:
        """
        Lightweight HTTP request to public Google Translate endpoint.
        Does not require external heavy packages or API keys.
        """
        try:
            encoded_text = urllib.parse.quote(text)
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target_lang}&dt=t&q={encoded_text}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            with urllib.request.urlopen(req, timeout=3.0) as response:
                if response.status == 200:
                    raw_json = response.read().decode('utf-8')
                    data = json.loads(raw_json)
                    # Result structure: [[["Translated text", "Original text", ...]]]
                    if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                        translated_parts = [part[0] for part in data[0] if part and len(part) > 0 and part[0]]
                        if translated_parts:
                            return "".join(translated_parts).strip()
        except Exception as e:
            logger.debug("Online translation fetch failed for text '%s' to '%s': %s", text, target_lang, e)
        return None

    def translate(self, text: str, target_language: str) -> dict:
        """
        Translates text to target_language.
        Returns:
            {
                "source_text": text,
                "target_language": target_lang_code,
                "translated_text": translated_string,
                "cached": bool,
                "fallback": bool
            }
        """
        if not text or not text.strip():
            return {
                "source_text": text or "",
                "target_language": target_language,
                "translated_text": text or "",
                "cached": False,
                "fallback": False
            }

        source_clean = text.strip()
        target_code = self._normalize_lang(target_language)

        # 1. English target language short-circuit
        if target_code == 'en' or target_code not in SUPPORTED_LANGUAGES:
            return {
                "source_text": source_clean,
                "target_language": target_code if target_code in SUPPORTED_LANGUAGES else 'en',
                "translated_text": source_clean,
                "cached": False,
                "fallback": False
            }

        cache_key = (source_clean.lower(), target_code)

        # 2. Check Session Cache
        with self._lock:
            if cache_key in self._cache:
                return {
                    "source_text": source_clean,
                    "target_language": target_code,
                    "translated_text": self._cache[cache_key],
                    "cached": True,
                    "fallback": False
                }

        # 3. Check Pre-compiled Canonical Phrase Dictionary
        normalized_phrase_key = source_clean.lower().replace('.', '').replace('?', '').replace('!', '').strip()
        if normalized_phrase_key in CANONICAL_PHRASE_DICTIONARY:
            lang_dict = CANONICAL_PHRASE_DICTIONARY[normalized_phrase_key]
            if target_code in lang_dict:
                translated = lang_dict[target_code]
                with self._lock:
                    self._cache[cache_key] = translated
                return {
                    "source_text": source_clean,
                    "target_language": target_code,
                    "translated_text": translated,
                    "cached": False,
                    "fallback": False
                }

        # 4. Online Translation Service
        online_result = self._online_google_translate(source_clean, target_code)
        if online_result:
            with self._lock:
                self._cache[cache_key] = online_result
            return {
                "source_text": source_clean,
                "target_language": target_code,
                "translated_text": online_result,
                "cached": False,
                "fallback": False
            }

        # 5. Graceful Fallback (Requirement 13: If translation fails, show original English sentence)
        logger.warning("Translation failed for '%s' to '%s', falling back to original text", source_clean, target_code)
        return {
            "source_text": source_clean,
            "target_language": target_code,
            "translated_text": source_clean,
            "cached": False,
            "fallback": True
        }

    def clear_cache(self):
        """Clears session translation cache."""
        with self._lock:
            self._cache.clear()

    def get_supported_languages(self) -> dict:
        """Returns dict of supported languages with BCP-47 codes."""
        return SUPPORTED_LANGUAGES


# Global singleton instance
_translator_instance = None
_translator_lock = threading.Lock()

def get_translator() -> MultilingualTranslator:
    """Returns the global MultilingualTranslator instance."""
    global _translator_instance
    if _translator_instance is None:
        with _translator_lock:
            if _translator_instance is None:
                _translator_instance = MultilingualTranslator()
    return _translator_instance

def translate(text: str, target_language: str) -> dict:
    """
    Convenience function for translation.
    Returns dict:
    {
        "source_text": "...",
        "target_language": "te",
        "translated_text": "..."
    }
    """
    return get_translator().translate(text, target_language)

