import json
from utils import read_json_file
from dotenv import load_dotenv
import os
import os.path as osp

class Data_Processor:
    def __init__(self):
        return
    
    def extract_conversations(self, data: list) -> list:
        '''
        Return a list of simulated conversations from the json data
        '''
        conversations = []

        for item in data:
            conversation = ""
            for turn in item['transcript']:
                conversation += f"{turn['type']}: {turn['data']}\n"
            
            conversations.append(conversation)

        return conversations

        
        
    
    
def main():
    load_dotenv()

    data_dir = os.getenv("DATA_DIR")

    data = read_json_file(osp.join(data_dir, 'sanitized-sample.json'))

    processer = Data_Processor()

    convs = processer.extract_conversations(data)
    print(convs)


    return 

if __name__ == '__main__':
    main()