import openai 
from dotenv import load_dotenv
import os
from utils import read_json_file, read_text_file
import json
import tiktoken

class OpenAI_LLM:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai
        self.model_type = os.getenv('DEFAULT_MODEL')

        return
    
    def get_response(self, instruction, user_input, output_format):
        # Choose encoding based on model type
        model_for_encoding = self.model_type if self.model_type else "gpt-3.5-turbo"
        try:
            encoding = tiktoken.encoding_for_model(model_for_encoding)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

        def count_tokens(text):
            return len(encoding.encode(text))

        response = self.client.chat.completions.create(
            model=self.model_type,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input}
            ],
            response_format=output_format,
            temperature=0.0
        )

        # Try to get token usage from response, fallback to tokenizer if not available
        try:
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
        except Exception:
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

        instruction_tokens = count_tokens(instruction)
        user_input_tokens = count_tokens(user_input)
        content = response.choices[0].message.content
        response_tokens = count_tokens(content)

        if output_format["type"] == "json_schema":
            response_obj = json.loads(content)
            return {
                "response": response_obj,
                "token_counts": {
                    "instruction_tokens": instruction_tokens,
                    "user_input_tokens": user_input_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens
                }
            }
        else:
            return {
                "response": content,
                "token_counts": {
                    "instruction_tokens": instruction_tokens,
                    "user_input_tokens": user_input_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens
                }
            }


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