from openai import OpenAI
import os
import json
from pydantic import BaseModel, Field
from typing import List
import requests 

RAW_DATA_DIR = 'raw_data'
ANNOTATED_DATA_DIR = 'annotated_data'
EXPERIMENTAL_DATA_DIR = 'prepared_experimental_data'

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = api_key)




class CharListResponse(BaseModel):
    participants: List[str] = Field(default_factory=list)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_chars(obj):
    return list(obj['mentions']['gold'])


def get_char_lists_annotated():
    ann_chars = {}
    
    for file_name in os.listdir(ANNOTATED_DATA_DIR):
        if not file_name.endswith('.json'):
            continue
        file_path = os.path.join(ANNOTATED_DATA_DIR, file_name) 
        obj = load_json(file_path)
        title = obj["title"]
        ann_chars[title] = extract_chars(obj)
    return ann_chars



def get_char_list_llm(text):
    is_ok, chars = send_char_list_prompt(text)
    if is_ok: 
        llm_char_list = chars
        reviewed_llm_char_list = send_review_prompt(chars, text)
    else: 
        print('Smth went wrong with the file')
        return [], []
    return llm_char_list, reviewed_llm_char_list





def send_char_list_prompt(text):

#The value <background_character> is used for characters who play a subordinate role in the text. They are part of the narrative world but do not fulfill any essential function for the plot or the development of the main characters. They are usually not referred to by a proper name in the text and are instead described by generic terms (e.g., "the woman", "the traveler", "the saleswoman") and mentioned only once or a few times.
 
    
    template = """
TEXT:
{text}

CHARACTERS:
A character is any entity in a fictional text that is explicitly represented as capable of agency, communication, thought, emotion, or perception. Characters may be human or anthropomorphized, artificial, zoomorphic, or supernatural beings.

SPECIAL CHARACTER VALUES:
Use <crowd> for collective entities such as groups, crowds, audiences, mobs, or teams when they participate in actions in the text.
If a named character acts as part of such a collective entity, include both the character’s name and <crowd>.
Use <background_character> as a single generic marker for unnamed minor characters who appear briefly and do not play a significant role in the plot.
Do not create multiple distinct <background_character> entries or attach descriptions/qualifiers to it.
<background_character> should appear at most once in the final list.

TASK:
Identify all entities that function as characters in the text.

CONSTRAINTS:
- Do not repeat characters.
- Use proper names whenever available, even if a generic descriptor exists.
- Use the most complete form explicitly stated in the text.
- Preserve titles such as Mr., Don, Dr., etc.
- Do not include locations, organizations, or objects unless they are explicitly personified or act as characters.
- Exclude religious, mythological, and abstract symbolic entities (e.g. God, Virgin Mary, saints, divine forces) unless they appear as physically acting characters within the fictional narrative.
- Invocation, speech references, prayers, or metaphors do not count as participation.

OUTPUT FORMAT:
Return the result in the required structured format.
"""

    complete_msg = [
        {
            'role': 'system',
            'content': 'You are a literature expert'
        },
        {
            'role': 'user',
            'content': template.format(text = text)
        }
    ]
    
    

    try: 
        response = client.beta.chat.completions.parse(
            model="gpt-5-mini",
            # model = "gpt-5.2",
            messages = complete_msg,
            response_format = CharListResponse
            
        )
        
        cleaned_response = response.choices[0].message.parsed.participants
        
        return True, cleaned_response
    
    except Exception as e:
        print(f'[X] API error: {e}')
        return False, []
        


def send_review_prompt(chars, txt):
    
    template = """
TEXT:
{text}

CURRENT_CHAR_LIST:
{characters}

TASK:
Review CURRENT_CHAR_LIST against the TEXT using the CHARACTER DEFINITION, SPECIAL CHARACTER VALUES, and CONSTRAINTS below.

You must:
- remove duplicates;
- merge alternative names that refer to the same character;
- use proper names whenever available, even if a generic descriptor also exists;
- if multiple references point to the same character, use the most complete proper name explicitly stated in the text;
- remove entities that are not characters;
- add missing valid characters only if they are explicitly supported by the TEXT;
- preserve special values such as <crowd> and <background_character> only when justified by the rules.

If CURRENT_CHAR_LIST already satisfies the CHARACTER DEFINITION, SPECIAL CHARACTER VALUES, and CONSTRAINTS, return it unchanged.
Do not change the list unless there is a clear reason based on the TEXT and the rules below.


CHARACTER DEFINITION:
A character is any entity in a fictional text that is explicitly represented as capable of agency, communication, thought, emotion, or perception. Characters may be human or anthropomorphized, artificial, zoomorphic, or supernatural beings.

SPECIAL CHARACTER VALUES:
Use <crowd> for collective entities such as groups, crowds, audiences, mobs, or teams when they participate in actions in the text.
If a named character acts as part of such a collective entity, include both the character’s name and <crowd>.
Use <background_character> as a single generic marker for unnamed minor characters who appear briefly and do not play a significant role in the plot.
Do not create multiple distinct <background_character> entries or attach descriptions/qualifiers to it.
<background_character> should appear at most once in the final list.

CONSTRAINTS:
- Do not repeat characters.
- Use proper names whenever available, even if a generic descriptor exists.
- Use the most complete form explicitly stated in the text.
- Preserve titles such as Mr., Don, Dr., etc.
- Do not include locations, organizations, or objects unless they are explicitly personified or act as characters.
- Exclude religious, mythological, and abstract symbolic entities (e.g. God, Virgin Mary, saints, divine forces) unless they appear as physically acting characters within the fictional narrative.
- Invocation, speech references, prayers, or metaphors do not count as participation.


OUTPUT FORMAT:
Return the result in the required structured format.

"""
    
    complete_msgs = [
        {
            'role': 'system',
            'content': 'You`re a literature reviewer'
        },
        {   'role': 'user',
             'content': template.format
                 (
                     text = txt,
                     characters = chars
                 )
        }
    ]
    
    response = client.beta.chat.completions.parse(
        #model = "gpt-5.2",
        model="gpt-5-mini",
        messages = complete_msgs,
        response_format = CharListResponse
        
    )
    
    reviewd_chars = response.choices[0].message.parsed.participants
    
    return reviewd_chars





def print_char_lists(dict_char_lists):
    for title, chars in dict_char_lists.items():
        chars.sort(key=str.lower)
        print(f'{title}:')
        for char in chars:
            print(f'  - {char}')
        

def get_phrase_text(phrase_spans, text): 
    phrase_text = []
    for span in phrase_spans:
        phrase_text.append(get_span_text(span, text))
    phrase_text = ' ... '.join(phrase_text)
    return phrase_text



def get_span_text(span, text):     # e.g. [24, 35]
    start, end = span
    return text[start:end]
    


def get_verby_phrases_and_sentences(text):
    response = requests.post("http://127.0.0.1:8000/segment", json={"text": text})
    #print(response.json())
    # Prints: {'verbal_phrases': [[[0, 30], [60, 69]], [[31, 47]], [[71, 90]]], 'sentences': [[0, 70], [71, 90]]}    
    dict_response = response.json()
    participations = []
    sentences = []
    
    for phrase_spans in dict_response['verbal_phrases']: # e.g. phrase_spans [[0, 30], [60, 69]]
        phrase_text = get_phrase_text(phrase_spans, text)    
        participations.append(
            {
                "spans": phrase_spans,
                "phrase_text": phrase_text,
                "agentive": [],
                "low_agentive": [],
                "passive": []
            }
        )
    for sent_span in dict_response['sentences']:
        sent_text = get_span_text(sent_span, text)
        sentences.append(
            {
                "span": sent_span,
                "sentence_text": sent_text,
            }
        )
    
    return participations, sentences





def get_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text 


def get_title_from_file(file_name):
    return os.path.splitext(file_name)[0]


def prepare_data(RAW_DATA_DIR):
    data = {}
    for file_name in os.listdir(RAW_DATA_DIR):
        if not file_name.endswith('.txt'):
            continue
        print(f'Processing file {file_name}...')
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        
        text = get_text_from_file(file_path)        # text
        title = get_title_from_file(file_name)      # title
        llm_char_list, _ = get_char_list_llm(text)    # getting char list via LLM
        participations, sentences = get_verby_phrases_and_sentences(text)    # getting phrases and sentences via verby
        
        data[title] = {
            "characters": sorted(llm_char_list, key = str.lower),
            "participations": participations,
            "text": text,
            "sentences": sentences
        }
        print(f"Successfully processed: {file_name}")
               
    return data


def save_data(data, EXPERIMENTAL_DATA_DIR):
    os.makedirs(EXPERIMENTAL_DATA_DIR, exist_ok = True)
    for title, item_data in data.items():
        output_path = os.path.join(EXPERIMENTAL_DATA_DIR, f'{title}_experimental.json') 
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(item_data, f, ensure_ascii = False, indent = 2)
            
    
    

def main():
    
    
    data = prepare_data(RAW_DATA_DIR)
    save_data(data, EXPERIMENTAL_DATA_DIR)
    
    
    
    
    
main()