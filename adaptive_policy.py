import numpy as np
from feature_extractor import Feature_Extractor
from utils import read_json_file

class LinearTSArm:
    def __init__(self, dim, prior_mean=None, prior_cov=None, lambda_reg=1.0):
        self.dim = dim  # Context dimension
        self.lambda_reg = lambda_reg  # Ridge regularization
        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(dim)
        self.prior_cov = prior_cov if prior_cov is not None else np.eye(dim)
        
        # Initialize posterior as prior
        self.mean = self.prior_mean.copy()
        self.cov = self.prior_cov.copy()
        self.inv_cov = np.linalg.inv(self.cov)  # For efficiency
    
        # Ensure shapes are correct
        assert self.mean.shape == (dim,), f"Mean shape {self.mean.shape} != ({dim},)"
        assert self.cov.shape == (dim, dim), f"Cov shape {self.cov.shape} != ({dim}, {dim})"
        assert self.inv_cov.shape == (dim, dim), f"Inv_cov shape {self.inv_cov.shape} != ({dim}, {dim})"


    def sample_theta(self):
        return np.random.multivariate_normal(self.mean, self.cov)

    def update(self, context, reward):
        # Ensure context matches dim
        if context.shape[0] != self.dim:
            raise ValueError(f"Context dimension {context.shape[0]} does not match expected {self.dim}")
        
        context = context.reshape(-1, 1)  # Convert to column vector
        # Update inverse covariance
        self.inv_cov += (context @ context.T) / self.lambda_reg
        # Recalculate covariance
        self.cov = np.linalg.inv(self.inv_cov)
        # Update mean
        temp = self.inv_cov @ self.mean.reshape(-1, 1) + (reward / self.lambda_reg) * context
        self.mean = (self.cov @ temp).flatten()

        # Verify shapes
        assert self.mean.shape == (self.dim,), f"Mean shape {self.mean.shape} != ({self.dim},)"
        assert self.cov.shape == (self.dim, self.dim), f"Cov shape {self.cov.shape} != ({self.dim}, {self.dim})"

class TSAdapter:
    def __init__(self, action_space, dim, lambda_reg=1.0, max_select=3):
        self.action_space = action_space  # Dict: {0: 'clarity', 1: 'active_listening', ...}
        self.num_actions = len(action_space)
        self.arms = [LinearTSArm(dim, lambda_reg=lambda_reg) for _ in range(self.num_actions)]
        self.prev_action_id = -1
        self.select_freq = 1
        self.max_select = max_select
        self.true_thetas = [np.random.randn(dim) for _ in range(len(action_space))]  # Hidden true params for simulation
        
        # init prev scores to calculate delta change
        self.prev_skill_score = 0
        self.prev_overall_score = 0


    def select_action(self, context):
        sampled_rewards = []
        
        for arm in self.arms:
            theta_sample = arm.sample_theta()
            sampled_rewards.append(np.dot(context, theta_sample))

        action_idx = np.argmax(sampled_rewards)

        # ensure no repeat
        if action_idx == self.prev_action_id:
            self.select_freq += 1
            if self.select_freq >= self.max_select:
                # select another action
                action_idx = np.argsort(sampled_rewards)[-2]
                self.select_freq = 1

        else:
            # reset counter
            self.select_freq = 1
        
        # update prev action id 
        self.prev_action_id = action_idx

        return action_idx, self.action_space[action_idx]  # Return index and name

    def update(self, action_idx, context, reward):
        self.arms[action_idx].update(context, reward)

    def calculate_reward_score(self, delta_skill_focus, delta_overall):
        # Simulate true linear component + proxy reward formula
        reward = 0.6 * delta_skill_focus + 0.4 * delta_overall
        return reward
    

class Baseline_Policy:
    def __init__(self):
        self.prev_selected_skill = ''
        self.select_freq = 1
        self.max_select = 3
        self.prev_skill_focus_score = 0
        self.prev_overall_score = 0
        return
    
    def select_action(self, feature_extractor: Feature_Extractor, simulation: dict) -> int:
        '''
        select the skill (feature) with the lowest score
        '''
        skill_scores = feature_extractor.extract_skill_scores(simulation)

        sorted_skills = sorted(skill_scores, key=skill_scores.get)
        
        weak_skill = sorted_skills[0]
        
        # ensure no freq repeat
        if weak_skill == self.prev_selected_skill:
            self.select_freq += 1
            if self.select_freq >= self.max_select:
                # select the second weakest skill
                weak_skill = sorted_skills[1]
                self.select_freq = 1
        else:
            # reset select frequent
            self.select_freq = 1

        # update prev select skill
        self.prev_selected_skill = weak_skill

        return weak_skill
        
    def calculate_reward_score(self, delta_skill_focus, delta_overall):
        reward = 0.6 * delta_skill_focus + 0.4 * delta_overall
        return reward

def true_reward_function(action_idx, context, true_thetas, delta_skill_focus, delta_overall, noise_std=0.1):
    # Simulate true linear component + proxy reward formula
    linear_part = np.dot(context, true_thetas[action_idx])
    reward = 0.6 * delta_skill_focus + 0.4 * delta_overall + linear_part + np.random.normal(0, noise_std)
    return reward


if __name__ == "__main__":
    np.random.seed(42)

    dim = 7

    action_space = {0: 'clarity and enthusiasm in pitch', 
                    1: 'active listening and objection handling', 
                    2: 'effective call to action', 
                    3: 'friendliness and respectful tone'}

    true_thetas = [np.random.randn(dim) for _ in range(len(action_space))]  # Hidden true params for simulation

    data = read_json_file('data/sanitized-sample.json')

    extractor = Feature_Extractor()

    ts_adapter = TSAdapter(action_space=action_space, dim=dim)

    prev_focus_skill_score = 0
    prev_overall_score = 0

    total_reward = 0

    for t, simulation in enumerate(data):
        baseline_feats = extractor.extract_baseline_feats(simulation)

        context = np.hstack(([baseline_feats[feat] for feat in baseline_feats]))

        action_idx, action_name = ts_adapter.select_action(context)

        print(f"Step {t}: Selected action '{action_name}' for context {context}")

        delta_skill_focus = baseline_feats[action_name] - prev_focus_skill_score

        delta_overall = baseline_feats["rubric_score"] - prev_overall_score

        reward = true_reward_function(action_idx, context, true_thetas, delta_skill_focus, delta_overall)

        ts_adapter.update(action_idx, context, reward)

        prev_overall_score = baseline_feats["rubric_score"]
        prev_focus_skill_score = baseline_feats[action_name]

        total_reward += reward



