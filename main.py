from openai import OpenAI
import os
import json
from pydantic import BaseModel, Field
from typing import List
import requests 
from itertools import combinations
from collections import defaultdict


#----------------------Global variables----------------------------------------
RAW_DATA_DIR = 'raw_data' # folder with the raw .txt files to process 
ANNOTATED_DATA_DIR = 'annotated_data' # folder with old annotated files (CATMA)
EXPERIMENTAL_DATA_DIR = 'prepared_experimental_data' # folder for output data

LABELS = ('agentive', 'low_agentive', 'passive')
CONTEXT_SENT_SIZE = 10 

MODEL_NAME = "gpt-5.2"

api_key = os.getenv("OPENAI_API_KEY") # taking gpt api-key from an environment variable
client = OpenAI(api_key = api_key)
#-----------------------------------------------------------------------------



class LabelsResponse(BaseModel):
    agentive: List[str] = Field(default_factory=list)
    low_agentive: List[str] = Field(default_factory=list)
    passive: List[str] = Field(default_factory=list)
    

class CharListResponse(BaseModel):
    participants: List[str] = Field(default_factory=list)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
        reviewed_llm_char_list = send_review_prompt(chars, text) #TODO: add error handling
    else: 
        print('Smth went wrong with the file')
        return [], []
    return llm_char_list, reviewed_llm_char_list





def send_char_list_prompt(text):

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
            model = MODEL_NAME,
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
        model = MODEL_NAME,
        messages = complete_msgs,
        response_format = CharListResponse
        
    )
    
    reviewed_chars = response.choices[0].message.parsed.participants
    
    return reviewed_chars



def send_labels_prompt(llm_char_list, context_text, phrase_text):
    template = """

CHARACTERS_LIST:

{llm_char_list}


CONTEXT_EXCERPT:
     
{context_text} 


TARGET_PHRASE:
     
"{phrase_text}"


DEFINITIONS:
     
CHARACTER ACTIONS:

All entities present in a text are considered characters if they are explicitly attributed the ability to act, communicate, or at least to think, feel and perceive in the text itself. 
These abilities are considered simultaneously as character actions. Character actions include intentional actions and non-intentional behaviors, speech or communication acts, but also internal events or feelings, and thus actions that are performed as well as imagined or unrealized actions.


SPECIAL CHARACTER VALUES:

The character <crowd> may be used to capture collective entities. If an identifiable character participates in the actions of a group of characters not named individually in the text (a collective entity), the corresponding text unit is assigned two values: once with the character’s name and once with the value <crowd>.
The value <background_character> is used for characters who play a subordinate role in the text. They are part of the narrative world but do not fulfill any essential function for the plot or the development of the main characters. They are usually not referred to by a proper name in the text and are instead described by generic terms (e.g., "the woman", "the traveler", "the saleswoman") and mentioned only once or a few times.


INSTRUCTIONS HOW TO CHOOSE A LABEL:

{labels_description}
 
 
TASK:
    
Given the full CHARACTERS_LIST of some text and one verbal phrase TARGET_PHRASE with its CONTEXT_EXCERPT,
select only those characters from the list who participate in this phrase,
according to the given definition of CHARACTER ACTIONS, 
and assign each from your selected characters to exactly one agentivity label.
The guidelines how to choose an agentivity label are given in INSTRUCTIONS HOW TO CHOOSE A LABEL above.

RESTRICTIONS:
CONTEXT_EXCERPT is provided only to resolve pronouns, ellipsis, and referring expressions in TARGET_PHRASE.
Do not assign labels based on actions, states, or events that occur only in CONTEXT_EXCERPT.
Only assign labels for participation in the action, state, or event expressed by TARGET_PHRASE itself.
Use ONLY exact character strings from CHARACTERS_LIST.
Most characters from CHARACTERS_LIST will not participate in TARGET_PHRASE.
Do not invent, rename, normalize, translate, or add characters.
Each selected character must appear in exactly one label list.
If no character fits a category, return an empty list for that category.
The mention does not necessarily have to be the character’s full name — it may also appear as a pronoun or any other referring expression.
Carefully follow the instructions and select strictly one label for a character.

OUTPUT FORMAT:
    
Return only the structured response with these fields: agentive, low_agentive, passive.
A character who does not participate in TARGET_PHRASE must not appear in any output list.
""".strip()

    labels_description = """
Important:
Do not decide labels based on grammatical voice (active/passive).
To assign the correct label (agentive, low_agentive, or passive), 
follow this step-by-step logic:

STEP 1:
Ask whether the action is **physically** performed or **physically** carried out by the figure associated with the annotation, or whether the capability for this action is attributed to them. 
Important: grammatical passive (e.g., “was discovered”) does NOT automatically mean the label "passive".
– If the figure does not perform the action and is only affected by it (or if the action by/for the figure is negated), assign the label: passive.
– If the figure performs the action (even if the agent is implicit), proceed to STEP 2.

STEP 2: 
If the action is not actually performed but only imagined, wished or hypothetical, assign the label low_agentive. Otherwise, proceed to STEP 3.

STEP 3:
Ask whether the action is consciously or intentionally caused, controlled, or influenced by the figure. 
Heuristic: if the text suggests that the character could stop or interrupt the activity, it counts as controlled.
– If yes, assign the label: agentive.
– If no or if control is not explicitly indicated, assign the label: low_agentive.
"""

    complete_msgs = [
        {   
            "role": "system",
            "content": "You are a linguistic annotator specializing in agentivity classification in literary texts."
        },
        {
            "role": "user",
            "content": template.format(
                llm_char_list = llm_char_list,
                context_text = context_text,
                phrase_text = phrase_text,
                labels_description = labels_description
                
            )
        }
    ]
    
    try: 
        response = client.beta.chat.completions.parse(
            model = MODEL_NAME,
            messages = complete_msgs,
            response_format = LabelsResponse
            
        )
        labels_obj = response.choices[0].message.parsed # this is an OBJECT of LabelsResponse 
        return labels_obj
    
    except Exception as e:
        print(f"[X] Labeling API error: {e}")
        return LabelsResponse() # then return default value which is []


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
    

def get_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text 


def get_title_from_file(file_name):
    return os.path.splitext(file_name)[0]


def save_data(data, EXPERIMENTAL_DATA_DIR):
    os.makedirs(EXPERIMENTAL_DATA_DIR, exist_ok = True)
    for title, item_data in data.items():
        output_path = os.path.join(EXPERIMENTAL_DATA_DIR, f'FINAL_{title}_experimental.json') 
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(item_data, f, ensure_ascii = False, indent = 2)


# the function returns ids of sentences that intersect given SPANS.
def get_sentences_ids_overlapping_spans(sentences, spans, min_overlap = 0):
    # find the minimum start and maximum end of all spans:
    min_span_start = min(span[0] for span in spans)
    max_span_end = max(span[1] for span in spans)
    
    sentence_ids_list = []
    #intersected = False
    for sent in sentences:
        #start and end of the current sentence:
        start_of_sentence = sent["span"][0]
        end_of_sentence = sent["span"][1]

        # If the beginning of a sentence is after the end of the rightmost span, then break the loop:
        if start_of_sentence > max_span_end:
            break  # ordered - there will be no further intersections

        # If the end of the sentence is before the start of the leftmost span - continue
        if end_of_sentence < min_span_start:
            continue

        #for a, b in mention_spans: #a, b - a pair of numbers (i.e., span). And so on for each span from SPANS
        for c, d in spans:
            overlap_start = max(start_of_sentence, c)
            overlap_end = min(end_of_sentence, d)
            if overlap_end - overlap_start >= min_overlap: #then intersected
                sentence_ids_list.append(sent["idx"]) #return spans of the sentences, not the sentences themselvs 
                break # current sentence has been already added, we have to go to the next one
    return sentence_ids_list
    

# given a phrase, building a context for that phrase (taking into account the global parametr CONTEXT_SENT_SIZE)
def get_context(phrase, sentences):
    phrase_spans = sorted(set(tuple(span) for span in phrase['spans'])) # из-за приколов с хэшируемыми объектами. Лист - не хешируемый, а тупл - да. Поэтому привели к виду тупл.
    phrase_spans = [
        list(span) 
        for span in phrase_spans # обратно в список преобразуем
    ] 
    
    phrase_sentences_ids = get_sentences_ids_overlapping_spans(sentences, phrase_spans) # get ids of sentences that the phrase intersects with
    phrase_sentences_ids.sort()
    min_id = phrase_sentences_ids[0]
    max_id = phrase_sentences_ids[-1]
    
    context_with_phrase_text = [
        sent['text'] 
        for sent in sentences[max(0, min_id - CONTEXT_SENT_SIZE) : min(len(sentences), max_id + CONTEXT_SENT_SIZE + 1)]
        
    ]
    
    return ' '.join(context_with_phrase_text)
    


def get_verby_phrases_and_sentences(text):
    response = requests.post("http://127.0.0.1:8000/segment", json={"text": text})
    dict_response = response.json()
    participations = []
    sentences = []
    
    for phrase_spans in dict_response['verbal_phrases']: # e.g. phrase_spans [[0, 30], [60, 69]]
        phrase_text = get_phrase_text(phrase_spans, text)    
        participations.append(
            {
                "spans": phrase_spans,
                "text": phrase_text,
                "agentive": [],
                "low_agentive": [],
                "passive": []
            }
        )
    for idx, sent_span in enumerate(dict_response['sentences']):
        sent_text = get_span_text(sent_span, text)
        sentences.append(
            {   
                "idx": idx,
                "span": sent_span,
                "text": sent_text,
            }
        )
    
    return participations, sentences



def get_labeled_phrases(reviewed_llm_char_list, participations, sentences):
    for p in participations:
        context = get_context(p, sentences)
        labels_response_obj = send_labels_prompt(reviewed_llm_char_list, context, p["text"])
        for label in LABELS: 
            p[label] = getattr(labels_response_obj, label) # because the attribute name is in a VARIABLE, which is a string
    return participations
   
    
# for now 1 scene = all the text
def get_scenes(sentences, text):
    scene_begin = 0
    scene_end = sentences[-1]["span"][1]
    
    return [[scene_begin, scene_end]]



def get_sentences_cooccurrence():
    return {}


def get_characters_interactions(participations):
    char_interactions = {}
    for p in participations:
        #getting "char" -> "labels of this char" dict (for current participation object)
        chars_agentivity_dict = defaultdict(list)
        for label in LABELS:
            for char in p[label]:
                chars_agentivity_dict[char].append(label)
        
        for a,b in combinations(chars_agentivity_dict.keys(), 2):
            char_1, char_2 = sorted((a, b))
            labels_1 = chars_agentivity_dict[char_1] # all labels of char 1 in CURRENT participation object
            labels_2 = chars_agentivity_dict[char_2] # all labels of char 2 in CURRENT participation object
            
            pair_key = f'{char_1} -- {char_2}'
            
            # If there is no such pair in the final dictionary yet, then we create a structure:
            if pair_key not in char_interactions:
                char_interactions[pair_key] = {
                    "num_interactions": 0,
                    "characters": [char_1, char_2],
                    "kinds": {}
                }
            char_interactions[pair_key]["num_interactions"] += 1
            for l1 in labels_1:
                for l2 in labels_2:
                    kind_1, kind_2 = sorted((l1, l2))
                    kind_key = f'{kind_1} -- {kind_2}'
                    if kind_key not in char_interactions[pair_key]["kinds"]:
                        char_interactions[pair_key]["kinds"][kind_key] = {
                            "count": 0,
                            "kinds": [kind_1, kind_2]
                        }
                    char_interactions[pair_key]["kinds"][kind_key]["count"] += 1
    
    return char_interactions
                

def get_characters_counts():
    return {}



def get_characters_agency(participations):
    characters_agency = {}
    for p in participations:
        for label in LABELS:
            for char in p[label]:
                if char not in characters_agency:
                    characters_agency[char] = {label: 0 for label in LABELS}                    
                characters_agency[char][label] += 1            
    return characters_agency



def get_individual_characters_agency(participations):
    ind_chars_agency = {}

    for p in participations:
        char_labels = [
            (char, label)   # here we construct tuples of the form: "(character, his label)"
            for label in LABELS
            for char in p[label]
        ]

        if len(char_labels) != 1:
            continue

        char, label = char_labels[0] # since what we got above is precisely a LIST. Of one element.

        if char not in ind_chars_agency:
            ind_chars_agency[char] = {label: 0 for label in LABELS}

        ind_chars_agency[char][label] += 1

    return ind_chars_agency

    



def get_character_graph(participations):
       sentences_cooccurrence = get_sentences_cooccurrence()  # for now we leave it empty
       characters_interactions = get_characters_interactions(participations)
       characters_counts = get_characters_counts()    # for now we leave it empty
       characters_agency = get_characters_agency(participations)
       individual_characters_agency = get_individual_characters_agency(participations)
       
       return [
           {
                "sentence_cooccurrence": sentences_cooccurrence,
                "character_interactions": characters_interactions,
                "character_counts": characters_counts,
                "character_agency": characters_agency,
                "individual_character_agency": individual_characters_agency
           }
       ]

       
def load_participations_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)    
    return obj["participations"]
  

  
def prepare_data(RAW_DATA_DIR):
    data = {}
    for file_name in os.listdir(RAW_DATA_DIR):
        if not file_name.endswith('.txt'):
            continue
        print(f'Processing file {file_name}...')
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        
        text = get_text_from_file(file_path)        # text
        title = get_title_from_file(file_name)      # title
        llm_char_list, reviewed_llm_char_list = get_char_list_llm(text)    # getting char list via LLM
        participations, sentences = get_verby_phrases_and_sentences(text)    # getting phrases (participations) and sentences via verby
        participations = get_labeled_phrases(reviewed_llm_char_list, participations, sentences) # make participations labeled (by LLM)
        
        #participations = load_participations_from_file('prepared_experimental_data/2_SMALL_Das Erdbeben in Chili_experimental.json')
        
        scenes = get_scenes(sentences, text) # getting "scenes" field
        character_graph = get_character_graph(participations) # getting "character_graph" field
        
        data[title] = {
            "characters": sorted(llm_char_list, key = str.lower),
            "title": title,
            "text": text,
            "participations": participations,
            "sentences": sentences,
            "scenes": scenes,
            "character_graph": character_graph
        }
        print(f"Successfully processed: {file_name}")
               
    return data

    


def main():
    data = prepare_data(RAW_DATA_DIR)
    save_data(data, EXPERIMENTAL_DATA_DIR)
    
    
    
if __name__ == '__main__':    
    main()
    
    
