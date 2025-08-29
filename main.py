from llm import OpenAI_LLM
from utils import read_json_file, read_text_file
from dotenv import load_dotenv
import os
import os.path as osp
from data_processer import Data_Processor
import json

def main():
    # load .env
    load_dotenv()

    data = read_json_file(osp.join(os.getenv("DATA_DIR"), 'sanitized-sample.json'))

    llm_feat_prompt = read_text_file(osp.join(os.getenv("PROMPT_DIR"), 'generate_llm_feats.txt'))
    llm_feat_res_format = read_json_file(osp.join(os.getenv("RESPONSE_FORMAT_DIR"), "generate_llm_feats.json"))

    data_processer = Data_Processor()

    conversations = data_processer.extract_conversations(data=data)

    llm = OpenAI_LLM()

    test = []
    
    for conversation in conversations[:5]:
        feats = llm.get_response(instruction = llm_feat_prompt, 
                                user_input = conversation,
                                output_format = llm_feat_res_format)
        
        test.append(feats)

    with open('test_llm_feats.json', 'w') as f:
        json.dump(test, f, indent=4)
    return

if __name__ == '__main__':
    main()