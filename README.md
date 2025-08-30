# Project Structure
The project is structured as below:
```
root
    |__artifacts (includes all output files)
    |__data (includes data file)
    |__prompts (includes all prompt files)
    |__response_formats (includes all files to define LLM's response formats)
    |__.env (environment file)
    |__app.log (log file includes all running logs and required metrics)
    |__ // other coding files

```

# Model Choices
- **LLM**: OpenAI gpt-4o-mini
- **Adaptive Policy**: Thompson Sampling

# How to Run Guidline
## Setup environment variables
In ```.env``` file, replace ```OPENAI_API_KEY``` with your API key. Other variables (e.g. folder and file paths) should be remained unchanged.

## Create conda environment
```
conda create -n {env_name}
```
## Active enviroment
```
conda activate {env_name}
```
##  Install requirements
```
pip install -r requirements.txt 
```
## Run the main file
```
python main.py
```
The ```main.py``` file implements the following tasks:
- Generate LLM features for each session (19 sessions)
- Run Thompson Sampling strategy to select 1 user's skill that need to be improved at each time step.
- Generate an adaptive coaching plan, which includes skill need to focus on, reason and next scenario stub.
- Calculate and compare metrics (e.g. reward, overall rubric score) between 2 strategies: **Thompson Sampling** vs **"Always weakest skills"**
- Using ```RandomForestRegressor``` to predict delta overall score between 2 consecutive time stamps. Evaluating the model using LOSO and R2 score.

The file also produces the following artifacts:
- **coaching_next.json**: include all information of the coaching plan for the next scenario
- **features.csv**: a DataFrame of all used features (numeric + LLM features) in all time stamps
- **token_counts.json**: detail token usage for each LLM request (e.g. generating LLM features, generating coaching plan)

Finally, the ```app.log``` file includes all running logs and required metrics.