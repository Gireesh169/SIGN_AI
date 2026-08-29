"""
SIGN-AI Vernacular Smart Classroom — Curriculum Assistant Module
Generates Foundational Literacy and Numeracy (FLN) lesson plans for Primary Education
(Class 1 to Class 5) in Hindi + Target Vernacular Language (Santhali, Ho, Mundari, English).
"""

import os
import sys
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translation_engine import translate

logger = logging.getLogger('sign_ai.curriculum_assistant')

CURRICULUM_DATABASE = {
    "class_1": {
        "fln_numeracy": [
            {
                "topic_id": "c1_num_1_10",
                "title_hi": "गिनती 1 से 10 (Counting 1 to 10)",
                "objective_hi": "1 से 10 तक की संख्याओं को पहचानना और गिनना सीखना।",
                "explanation_hi": "आज हम 1 से 10 तक गिनती सीखेंगे। हमारे दो हाथों में कुल 10 उंगलियां होती हैं।",
                "activity_hi": "अपने हाथों की उंगलियां गिनें और संख्या बोलें।",
                "assessment_hi": "1, 2, __, 4 में छूटी हुई संख्या बताएं।",
                "vocab": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "water", "tree"]
            },
            {
                "topic_id": "c1_num_shapes",
                "title_hi": "मूल आकृतियाँ (Basic Shapes)",
                "objective_hi": "गोल (वृत्त) और चौकोर (वर्ग) आकृतियों की पहचान करना।",
                "explanation_hi": "सूरज और गेंद गोल होती हैं। आपकी किताब चौकोर होती है।",
                "activity_hi": "कक्षा में गोल वस्तुओं को ढूंढें।",
                "assessment_hi": "गेंद की आकृति कैसी होती है?",
                "vocab": ["sun", "book", "school"]
            }
        ],
        "fln_literacy": [
            {
                "topic_id": "c1_lit_greetings",
                "title_hi": "अभिवादन और परिचय (Greetings & Introduction)",
                "objective_hi": "कक्षा में एक-दूसरे का स्वागत करना और नाम बताना।",
                "explanation_hi": "जब हम सुबह मिलते हैं, तो हम एक-दूसरे को नमस्ते कहते हैं।",
                "activity_hi": "अपने साथी से उसका नाम पूछें।",
                "assessment_hi": "सुबह मिलने पर क्या कहते हैं?",
                "vocab": ["good morning", "well done", "thank you"]
            }
        ]
    },
    "class_3": {
        "fln_numeracy": [
            {
                "topic_id": "c3_num_addition",
                "title_hi": "जोड़ और घटाव (Addition & Subtraction)",
                "objective_hi": "दो अंकों की संख्याओं को जोड़ना।",
                "explanation_hi": "यदि आपके पास 5 सेब हैं और दोस्त आपको 3 सेब और देता है, तो आपके पास कुल 8 सेब होंगे।",
                "activity_hi": "कंकड़ या तिल्लियों की मदद से 5 + 3 गिनें।",
                "assessment_hi": "5 + 3 = कितना?",
                "vocab": ["5", "3", "8", "apple"]
            }
        ]
    }
}

def generate_bilingual_lesson(grade: str = "class_1", subject: str = "fln_numeracy", topic_id: str = "c1_num_1_10", target_language: str = "sat") -> dict:
    """
    Generates a complete bilingual lesson plan with original Hindi and translated target language.
    """
    grade_key = grade.lower().replace(" ", "_")
    if grade_key not in CURRICULUM_DATABASE:
        grade_key = "class_1"
        
    subj_dict = CURRICULUM_DATABASE[grade_key]
    subj_lessons = subj_dict.get(subject, list(subj_dict.values())[0])
    
    # Find matching topic or default to first
    topic_data = subj_lessons[0]
    for lesson in subj_lessons:
        if lesson["topic_id"] == topic_id:
            topic_data = lesson
            break
            
    # Translate parts into target language
    title_tr = translate(topic_data["title_hi"], "hi", target_language)
    obj_tr = translate(topic_data["objective_hi"], "hi", target_language)
    exp_tr = translate(topic_data["explanation_hi"], "hi", target_language)
    act_tr = translate(topic_data["activity_hi"], "hi", target_language)
    ass_tr = translate(topic_data["assessment_hi"], "hi", target_language)
    
    vocab_bilingual = []
    for item in topic_data.get("vocab", []):
        v_tr = translate(item, "en", target_language)
        v_hi = translate(item, "en", "hi")
        vocab_bilingual.append({
            "key": item,
            "hindi": v_hi["translated_text"],
            "target": v_tr["translated_text"]
        })

    return {
        "status": "success",
        "grade": grade_key,
        "subject": subject,
        "topic_id": topic_data["topic_id"],
        "target_language": target_language,
        "lesson_plan": {
            "title": {
                "hindi": topic_data["title_hi"],
                "target": title_tr["translated_text"]
            },
            "learning_objective": {
                "hindi": topic_data["objective_hi"],
                "target": obj_tr["translated_text"]
            },
            "lesson_explanation": {
                "hindi": topic_data["explanation_hi"],
                "target": exp_tr["translated_text"]
            },
            "activity_instructions": {
                "hindi": topic_data["activity_hi"],
                "target": act_tr["translated_text"]
            },
            "assessment_prompt": {
                "hindi": topic_data["assessment_hi"],
                "target": ass_tr["translated_text"]
            },
            "bilingual_vocabulary": vocab_bilingual
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def list_curriculum_options() -> dict:
    """Returns available grades, subjects, and topics."""
    return {
        "grades": [
            {"id": "class_1", "name": "Class 1 (Primary FLN)"},
            {"id": "class_2", "name": "Class 2 (Primary FLN)"},
            {"id": "class_3", "name": "Class 3 (Primary FLN)"},
            {"id": "class_4", "name": "Class 4 (Primary)"},
            {"id": "class_5", "name": "Class 5 (Primary)"}
        ],
        "subjects": [
            {"id": "fln_numeracy", "name": "Foundational Numeracy (गणित)"},
            {"id": "fln_literacy", "name": "Foundational Literacy (भाषा और साक्षरता)"},
            {"id": "evs", "name": "Environmental Studies (पर्यावरण अध्ययन)"}
        ],
        "curriculum_db": CURRICULUM_DATABASE
    }

if __name__ == '__main__':
    print("Testing Vernacular Curriculum Assistant...")
    lesson = generate_bilingual_lesson("class_1", "fln_numeracy", "c1_num_1_10", "sat")
    print("Generated Santhali Lesson:", json.dumps(lesson, indent=2, ensure_ascii=False))
