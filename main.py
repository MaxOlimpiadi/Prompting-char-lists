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
        else: 
            print(f'Smth went wrong with the file {title}')
            total_failed += 1
    
    print(f'  [X] TOTAL FAILED: {total_failed}')
    print(f'  [X] TOTAL SUCCES: {total_success}')
    
    return llm_char_lists





def send_char_list_prompt(text):
    
    template = """
TEXT:
{text}

TASK:
Identify all characters mentioned in the text. 

CONSTRAINTS:
- Do not repeat characters.
- Use the most complete form of the name.
- Preserve titles such as Mr., Don, Dr., etc.
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
        



def print_char_lists(dict_char_lists):
    for title, chars in dict_char_lists.items():
        print(f'{title}:')
        for char in chars:
            print(f'  - {char}')
        
    

    

def main():
    
    ann_chars = get_char_lists_annotated()
    print_char_lists(ann_chars)
    
    print('\n\n------------------LLM CHAR LISTS--------------------\n\n')
    
    llm_chars = get_char_lists_llm()
    print_char_lists(llm_chars)
        
    
    
main()