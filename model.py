import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
def load_data():
    psg_data = pd.read_csv('../data/psg_500_matches.csv')
    barcelona_data = pd.read_csv('../data/barcelona_500_matches.csv')
    return psg_data, barcelona_data

# Combine datasets and prepare training data
def prepare_data(psg_data, barcelona_data):
    combined_data = pd.concat([psg_data, barcelona_data], ignore_index=True)
    barcelona_vs_psg = combined_data[(combined_data['team'] == 'Barcelona') & 
                                     (combined_data['opponent_form'] > 0)]
    X = barcelona_vs_psg[['team_form', 'opponent_form']]
    y = barcelona_vs_psg['result']
    return X, y

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, accuracy, report

# Main function
def main():
    psg_data, barcelona_data = load_data()
    X, y = prepare_data(psg_data, barcelona_data)
    model, accuracy, report = train_model(X, y)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n")
    print(report)

if __name__ == "__main__":
    main()
