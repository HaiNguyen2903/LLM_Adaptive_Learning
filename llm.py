import openai 
from dotenv import load_dotenv
import os
from utils import read_json_file, read_text_file
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

    llm = OpenAI_LLM()
    # print(llm.client)

    # print(os.getenv('t'))

    name = 'generate_adapt_coach'

    # format = read_json_file('response_formats/generate_coaching_card.json')
    # prompt = read_text_file('prompts/generate_coaching_card.txt')

    format = read_json_file(f'response_formats/{name}.json')
    prompt = read_text_file(f'prompts/{name}.txt')

    inputs = """
    {
        "llm_persona_personality_traits": {
            "openness": 6,
            "neuroticism": 3,
            "extraversion": 5,
            "agreeableness": 5,
            "conscientiousness": 9
        },
        "weak_skill": "active listening",
        "current_score": 50,
    }
    """
    
    res = llm.get_response(instruction=prompt, user_input=inputs, output_format=format)
    print(res)

    return 

if __name__ == '__main__':
    main()