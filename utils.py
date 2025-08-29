import json

def read_text_file(path):
    with open(path, 'r') as f:
        data = f.read()

    return data

def read_json_file(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    return data