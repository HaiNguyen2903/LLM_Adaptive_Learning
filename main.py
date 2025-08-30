from llm import OpenAI_LLM
from utils import read_json_file, read_text_file
from dotenv import load_dotenv
import os
import os.path as osp
from data_processer import Data_Processor
from feature_extractor import Feature_Extractor
from adaptive_policy import TSAdapter, Baseline_Policy
from adaptive_coach import AdaptiveCoacher
import numpy as np
import json
import logging
from IPython import embed
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Custom formatter for structured log output
class CustomFormatter(logging.Formatter):
    # Define log format with timestamp, level, and message
    format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    def __init__(self):
        super().__init__(fmt=self.format_str, datefmt='%Y-%m-%d %H:%M:%S')

def setup_logging(logger_name="my_app", log_level=logging.INFO, log_file="app.log"):
    """
    Configure logging to save output to a file with custom formatting.
    
    Args:
        logger_name (str): Name of the logger
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file (str): Path to the log file
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Clear any existing handlers to avoid duplicate logs
    logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(log_level)
    
    # Apply custom formatter
    file_handler.setFormatter(CustomFormatter())
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger

def load_prompt_with_schema(prompt_dir, format_dir, filename):
    prompt = read_text_file(osp.join(prompt_dir, f'{filename}.txt'))
    format = read_json_file(osp.join(format_dir, f"{filename}.json"))

    return prompt, format

def log_session(logger, timestep, used_baseline_features, used_llm_features, 
                reward_llm, delta_overall_perent_llm,
                reward_baseline, delta_overall_perent_baseline):
    logger.info(f"Session: {timestep+1} | Step: {timestep}")
    logger.info(f"Features: Baseline: {[feat for feat in used_baseline_features]} | LLM: {[feat for feat in used_llm_features]}")
    logger.info(f"Policy: Thompson Sampling | Reward Function: 0.6*Δskill + 0.4*Δoverall")
    logger.info(f"Results: Baseline (weakest skill): Reward: {reward_baseline} | % Delta Overall: {delta_overall_perent_baseline}%")
    logger.info(f"Results: Thompson Sampling: Reward: {reward_llm} | % Delta Overall: {delta_overall_perent_llm}%\n")

    return

def run_tsadapter(ts_adapter, baseline_feats, context):
    action_id, action_name = ts_adapter.select_action(context=context)

    # print(f"Selected action '{action_name}' for context {context}")

    delta_skill_focus = baseline_feats[action_name] - ts_adapter.prev_skill_score
    delta_overall = baseline_feats["rubric_score"] - ts_adapter.prev_overall_score
    
    delta_overall_perent = round(delta_overall / (ts_adapter.prev_overall_score+1), 2) * 100

    reward = ts_adapter.calculate_reward_score(delta_skill_focus=delta_skill_focus,
                                            delta_overall=delta_overall)
    
    # update distribution
    ts_adapter.update(action_id, context, reward)

    # update previous scores 
    ts_adapter.prev_skill_score = baseline_feats[action_name]
    ts_adapter.prev_overall_score = baseline_feats["rubric_score"]
    
    return reward, delta_overall_perent, action_name

def run_baseline_policy(baseline_policy: Baseline_Policy, feature_extractor: Feature_Extractor, 
                        baseline_features: dict, simulation: dict):
    weak_skill = baseline_policy.select_action(feature_extractor=feature_extractor, simulation=simulation)

    delta_skill_focus = baseline_features[weak_skill] - baseline_policy.prev_skill_focus_score
    delta_overall = baseline_features["rubric_score"] - baseline_policy.prev_overall_score

    delta_overall_perent = round(delta_overall / (baseline_policy.prev_overall_score+1), 2) * 100

    reward = baseline_policy.calculate_reward_score(delta_skill_focus=delta_skill_focus, 
                                                    delta_overall=delta_overall)
    
    # update previous score
    baseline_policy.prev_skill_focus_score = baseline_features[weak_skill]
    baseline_policy.prev_overall_score = baseline_features["rubric_score"]
    
    return reward, delta_overall_perent

def predict_and_evaluate(df_features, delta_overall_scores):
    X = df_features
    y = np.array(delta_overall_scores)

    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        y_true.extend(y_test)
        y_pred.extend(pred)

    r2 = r2_score(y_true, y_pred)

    return r2

def main():
    # load .env
    load_dotenv()

    # init logger
    logger = setup_logging(logger_name="", log_level=logging.INFO, log_file="app.log")

    '''
    Init env vars
    '''
    DATA_DIR = os.getenv("DATA_DIR")
    PROMPT_DIR = os.getenv("PROMPT_DIR")
    RESPONSE_FORMAT_DIR = os.getenv("RESPONSE_FORMAT_DIR")
    ARTIFACT_DIR = os.getenv("ARTIFACT_DIR")

    '''
    Init data & LLM variables
    '''
    data = read_json_file(osp.join(DATA_DIR, 'sanitized-sample.json'))
    
    feature_extractor = Feature_Extractor()
    
    llm = OpenAI_LLM()

    '''
    Init Thompson Sampling
    '''
    # init action space for Thompson Sampling
    action_space = {0: 'clarity and enthusiasm in pitch', 
                    1: 'active listening and objection handling', 
                    2: 'effective call to action', 
                    3: 'friendliness and respectful tone'}

    # define used features
    used_baseline_features = ["rubric_score", "talk_ratio", "latency"]
    
    used_llm_features = [
        "Objection mirroring",
        "Question ratio",
        "Empathy markers",
        "Hedging intensity",
        "You/We orientation",
        "Pushiness vs. collaborative tone"
    ]

    # adapter using addition LLM features
    ts_adapter = TSAdapter(action_space=action_space, dim=len(used_baseline_features) + len(used_llm_features))

    # baseline policy (select weakest skill)
    baseline_policy = Baseline_Policy()


    '''
    Init prompts and schemas 
    '''
    # For generating llm features
    llm_feat_prompt, llm_feat_format = load_prompt_with_schema(prompt_dir=PROMPT_DIR,
                                                                format_dir=RESPONSE_FORMAT_DIR,
                                                                filename="generate_llm_feats")
    
    # For generating adaptive coaching plan
    coach_plan_prompt, coach_plan_format = load_prompt_with_schema(prompt_dir=PROMPT_DIR,
                                                                   format_dir=RESPONSE_FORMAT_DIR,
                                                                   filename="generate_adapt_coach")
    
    '''
    Init vars for outputs
    '''
    all_features = []
    coaching_next = {}
    delta_overall_scores = []

    
    '''
    Simulate time serie data
    '''

    prev_overall_score = 0

    for t, simulation in enumerate(data[:2]):
        '''
        Run Thompson Sampling
        '''
        # calculate baseline features and baseline context
        baseline_feats = feature_extractor.extract_baseline_feats(simulation)
 
        baseline_context = np.hstack([baseline_feats[feat] for feat in baseline_feats if feat in used_baseline_features])

        # calculate llm features and addition llm context (baseline + LLM)
        llm_feats = feature_extractor.extract_llm_feats(simulation=simulation, llm=llm,
                                                        instruction=llm_feat_prompt, 
                                                        output_format=llm_feat_format)
        
        # save output
        session_feats = {
            feat: baseline_feats[feat] for feat in baseline_feats if feat in used_baseline_features
        }

        for feat in llm_feats["features"]:
            if feat["name"] in used_llm_features:
                session_feats[feat["name"]] = feat["score"]

        all_features.append(session_feats)
        
        
        # filter llm features
        llm_feats = [feat["score"] for feat in llm_feats["features"] if feat["name"] in used_llm_features]
        
        
        # combine baseline with llm
        additon_llm_context = np.hstack((baseline_context, np.array(llm_feats)))

        # run tsadapter
        reward_llm, delta_overall_perent_llm, weak_skill = run_tsadapter(ts_adapter=ts_adapter,
                                                            baseline_feats=baseline_feats,
                                                            context=additon_llm_context)
        
        '''
        Run baseline policy
        '''
        # run tsadapter
        reward_baseline, delta_overall_perent_baseline = run_baseline_policy(
            baseline_policy=baseline_policy,
            feature_extractor=feature_extractor,
            baseline_features=baseline_feats,
            simulation=simulation
        )

        # save session overall score
        delta_overall_scores.append(baseline_feats["rubric_score"] - prev_overall_score)
        prev_overall_score = baseline_feats["rubric_score"]

        '''
        Generate Coaching plan
        '''
        input = {
            "llm_persona_personality_traits": simulation["persona_data"]["simulation_persona"]["public_persona_info"]["persona_personality_traits"],
            "weak_skill": weak_skill,
            "current_score": baseline_feats[weak_skill]
        }

        coach_plan = llm.get_response(instruction=coach_plan_prompt,
                                      user_input=str(input),
                                      output_format=coach_plan_format)
        
        coaching_next[t] = coach_plan
        
        '''
        Logging
        '''
        log_session(timestep=t, logger=logger,
                 used_baseline_features=used_llm_features,
                 used_llm_features=used_llm_features,
                 reward_llm=reward_llm,
                 delta_overall_perent_llm = delta_overall_perent_llm,
                 reward_baseline=reward_baseline,
                 delta_overall_perent_baseline=delta_overall_perent_baseline)
        

    """
    save output
    """
    df_all_feats = pd.DataFrame(all_features)
    df_all_feats.to_csv(osp.join(ARTIFACT_DIR, 'features.csv'), index=False)

    with open(osp.join(ARTIFACT_DIR, 'coaching_next.json'), 'w') as f:
        json.dump(coaching_next, f)

    """
    predict delta overall score and evaluate
    """
    df_baseline_feats = df_all_feats[used_baseline_features]

    r2_baseline = predict_and_evaluate(df_features=df_baseline_feats, delta_overall_scores=delta_overall_scores)
    r2_all_feats = predict_and_evaluate(df_features=df_all_feats, delta_overall_scores=delta_overall_scores)
    

    print(r2_all_feats, r2_baseline)

    logger.info(f"Ablation: Baseline: R2 = {r2_baseline}")
    logger.info(f"Ablation: With LLM Features: R2 = {r2_all_feats}")

if __name__ == '__main__':
    main()