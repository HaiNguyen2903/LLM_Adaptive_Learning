import openai 
from dotenv import load_dotenv
import os
from utils import read_json_file
import json

class OpenAI_LLM:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai
        self.model_type = os.getenv('DEFAULT_MODEL')

        return
    
    def get_response(self, instruction, user_input, output_format):    
        response = self.client.chat.completions.create(
            model=self.model_type,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input}
            ],
            response_format=output_format,
            temperature=0.0
        )

        if output_format["type"] == "json_schema":
            return json.loads(response.choices[0].message.content)

        return response.choices[0].message.content


def main():
    load_dotenv()

    # llm = OpenAI_LLM()
    # print(llm.client)

    # print(os.getenv('t'))

    format = read_json_file('response_formats/generate_llm_feats.json')

    return 

if __name__ == '__main__':
    main()