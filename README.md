# Match-Predictor

An accurate predictor of Football Matches of our club. You can predict your matches and get to your flag
Your Flag format is flag{accuracy percentage,precision,recall,dataset samples no in each,Nos of parameter}

## Dataset

Place the following datasets in the `data/` folder:
- `psg_500_matches.csv`
- `barcelona_500_matches.csv`

## Features
- `team_form`: Form rating of the team.
- `opponent_form`: Form rating of the opponent.
- `goal_difference`: Difference in scores (team - opponent).

## Model
The project uses a Random Forest classifier, fine-tuned with GridSearchCV.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the training script:

bash
python src/train_model.py



