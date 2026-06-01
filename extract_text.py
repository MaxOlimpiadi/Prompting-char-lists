# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:03:54 2026

@author: zidan
"""

import os
import json

ANNOTATED_DATA_FOLDER = 'annotated_data'
OUTPUT_FOLDER = 'raw_data'


    

def main():
    for file_name in os.listdir(ANNOTATED_DATA_FOLDER):
        file_path = os.path.join(ANNOTATED_DATA_FOLDER, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
            text = obj['text']
        base_name = os.path.splitext(file_name)[0] 
        output_path = os.path.join(OUTPUT_FOLDER, f'{base_name}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    
main()