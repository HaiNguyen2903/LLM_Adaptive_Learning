from utils import read_json_file, read_text_file
from datetime import datetime
from llm import OpenAI_LLM
from dotenv import load_dotenv
import numpy as np

class Feature_Extractor:
    def __init__(self):
        return
    
    def _extract_conversation(self, simulation) -> list:
        '''
        Return a list of simulated conversations from the json data
        '''

        conversation = ""
        for turn in simulation['transcript']:
            conversation += f"{turn['type']}: {turn['data']}\n"
            
        return conversation
    
    def _time_difference_in_seconds_iso(self, timestamp1_str: str, timestamp2_str: str) -> dict:
        """
        Calculate the difference between two ISO 8601 timestamps in seconds.
        Args:
            timestamp1_str (str): First timestamp (e.g., "2025-08-09T13:20:04.638Z")
            timestamp2_str (str): Second timestamp (e.g., "2025-08-09T13:17:05.053Z")
        Returns:
            float: Difference in seconds (positive if timestamp1 > timestamp2, negative otherwise)
        """
        from datetime import datetime, timezone

        # Remove 'Z' and parse as UTC
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        t1 = datetime.strptime(timestamp1_str.rstrip('Z'), fmt).replace(tzinfo=timezone.utc)
        t2 = datetime.strptime(timestamp2_str.rstrip('Z'), fmt).replace(tzinfo=timezone.utc)
        return (t1 - t2).total_seconds()

    def extract_baseline_feats(self, simulation):
        # overall score
        overall_score = simulation["assessment_data"]["overall"]["score"]

        # talk ratio
        transcript = simulation['transcript']
        user_turns = sum([1 if turn['type'] == 'user' else 0 for turn in transcript])
        talk_ratio = user_turns / len(transcript)

        # latency
        latencies = []
        prev_timestamp = transcript[0]["timestamp"]

        for turn in transcript[1:]:
            if turn["type"] == "user":
                curr_timestamp = turn["timestamp"]
                latencies.append(self._time_difference_in_seconds_iso(curr_timestamp, prev_timestamp))
                # update timestamp
                prev_timestamp = curr_timestamp   

        latency = sum(latencies) / len(latencies)

        skill_scores = self.extract_skill_scores(simulation)

        feats = {
            "rubric_score": overall_score,
            "talk_ratio": talk_ratio,
            "latency": latency
        }

        for skill in skill_scores:
            feats[skill] = skill_scores[skill]

        return feats

        
        
    def extract_llm_feats(self, llm: OpenAI_LLM, instruction: str, output_format: dict, simulation: dict) -> dict:
        conversation = self._extract_conversation(simulation)
        
        response = llm.get_response(instruction=instruction, 
                                     user_input=conversation,
                                     output_format=output_format)
        
        llm_feats = response["response"]
        token_counts = response["token_counts"]
        
        return llm_feats, token_counts
    
    def extract_skill_scores(self, simulation: dict) -> np.array:
        scores = {}

        criteria = simulation["assessment_data"]["criteria"]

        for name in criteria:
            scores[name] = criteria[name]["score"]

        return scores

        

def main():
    load_dotenv()

    data = read_json_file('data/sanitized-sample.json')

    extractor = Feature_Extractor()

    llm = OpenAI_LLM()

    instruction = read_text_file('prompts/generate_llm_feats.txt')
    output_format = read_json_file('response_formats/generate_llm_feats.json')

    # feats = extractor.extract_llm_feats(llm=llm, instruction=instruction, output_format=output_format)

    print(extractor.extract_baseline_feats(data[1]))

    return 

if __name__ == '__main__':
    main()