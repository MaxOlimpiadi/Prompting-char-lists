from openai import OpenAI
import os
import json
from pydantic import BaseModel, Field
from typing import List

RAW_DATA_DIR = 'raw_data'
ANNOTATED_DATA_DIR = 'annotated_data'

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



def get_char_lists_llm():
    total_success = 0
    total_failed = 0
    llm_char_lists = {}
    reviewed_llm_char_lists = {}
    for file_name in os.listdir(RAW_DATA_DIR):
        if not file_name.endswith('.txt'):
            continue
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        txt = load_text(file_path)
        title = os.path.splitext(file_name)[0]
        is_ok, chars = send_char_list_prompt(txt)
        if is_ok: 
            total_success += 1
            llm_char_lists[title] = chars
            reviewed_chars = send_review_prompt(chars, txt)
            reviewed_llm_char_lists[title] = reviewed_chars
        else: 
            print(f'Smth went wrong with the file {title}')
            total_failed += 1
    
    print(f'  [X] TOTAL FAILED: {total_failed}')
    print(f'  [X] TOTAL SUCCES: {total_success}\n\n')
    
    return llm_char_lists, reviewed_llm_char_lists





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
            model = "gpt-5.2",
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
        model = "gpt-5.2",
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
        
    

    

def main():
    
    print('\n\n------------------GOLD CHAR LISTS--------------------\n\n')
    ann_chars = get_char_lists_annotated()
    print_char_lists(ann_chars)
    
    print('\n\n------------------LLM CHAR LISTS--------------------\n\n')
    
    llm_char_lists, reviewed_llm_char_lists = get_char_lists_llm()
    print('\n\n+++++Raw LLM char lists:+++++\n')
    print_char_lists(llm_char_lists)
    
    print('\n\n+++++Reviewed LLM char lists:+++++\n')
    print_char_lists(reviewed_llm_char_lists)
    
main()