from llm import OpenAI_LLM
from utils import read_json_file, read_text_file
from dotenv import load_dotenv

class AdaptiveCoacher:
    def __init__(self):
        return
    
    def generate_coaching_plan(self, llm: OpenAI_LLM, instruction: str, user_input: dict, 
                               output_format: dict) -> dict:
        response = llm.get_response(instruction=instruction, user_input=user_input,
                                      output_format=output_format)
        
        coach_plan = response["response"]
        token_counts = response["token_counts"]
        
        return coach_plan, token_counts
    

def main():
    load_dotenv()

    llm = OpenAI_LLM()

    coacher = AdaptiveCoacher()

    name = "generate_adapt_coach"

    format = read_json_file(f'response_formats/{name}.json')
    prompt = read_text_file(f'prompts/{name}.txt')

    inputs = {
        "llm_persona_personality_traits": {
            "openness": 6,
            "neuroticism": 3,
            "extraversion": 5,
            "agreeableness": 5,
            "conscientiousness": 9
        },
        "weak_skill": "active listening",
        "current_score": 50,
        "reason": "Great counter-arguments, but enhance active listening and objection exploration."
    }

    plan = coacher.generate_coaching_plan(llm, instruction=prompt, 
                                          user_input=str(inputs), output_format=format)
    
    print(plan)

    return

if __name__ == '__main__':
    main()

