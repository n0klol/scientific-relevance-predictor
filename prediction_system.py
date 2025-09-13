from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pickle
import pandas as pd
import os

DATABASE_PATH = "database.tsv"

MODELS_DIR = "models"

AVAILABLE_MODELS = {
    "1": ("LinearSVC", os.path.join(MODELS_DIR, "linear_svc.pkl")),
    "2": ("SGD-Logistic", os.path.join(MODELS_DIR, "sgd_logistic.pkl")),
    "3": ("ComplementNB", os.path.join(MODELS_DIR, "complement_nb.pkl")),
    "4": ("RidgeClassifier", os.path.join(MODELS_DIR, "ridge_classifier.pkl")),
}

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path, sep="\t")
        if "text" not in df.columns or "link_accessibility" not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'link_accessibility' columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def get_dataset():
    print("\n=== Dataset Options ===")
    print("1. Use existing test dataset from database")
    print("2. Import new test dataset into database")

    choice = input("Choose option (1 or 2): ").strip()

    if choice == "1":
        if os.path.exists(DATABASE_PATH):
            print(f"\nLoading dataset from {DATABASE_PATH}...")
            return load_dataset(DATABASE_PATH)
        else:
            print("\nNo database file found. You must import a dataset first.")
            return get_dataset()

    elif choice == "2":
        Tk().withdraw()
        file_path = askopenfilename(
            title="Select Test Dataset",
            filetypes=[("TSV files", "*.tsv"), ("All files", "*.*")]
        )
        if not file_path:
            print("No file selected. Exiting...")
            return None
        else:
            df = load_dataset(file_path)
            if df is not None:
                # Save a copy to simulate local database
                df.to_csv(DATABASE_PATH, sep="\t", index=False)
                print(f"\nDataset imported and saved to {DATABASE_PATH}")
            return df
    else:
        print("Invalid choice. Try again.")
        return get_dataset()


def choose_row(df):
    print(f"\nDataset loaded with {len(df)} rows.")

    while True:
        try:
            idx = int(input("Enter the row index you want to use for prediction: "))
            if 0 <= idx < len(df):
                text = df.iloc[idx]["text"]
                print(f"\nRow {idx} selected:\n{text}\n")
                return text
            else:
                print("Invalid index, try again.")
        except ValueError:
            print("Please enter a valid integer.")


def choose_model():
    print("\nAvailable models:")
    for key, (name, _) in AVAILABLE_MODELS.items():
        print(f"{key}. {name}")

    while True:
        choice = input("Choose a model (1-4): ").strip()
        if choice in AVAILABLE_MODELS:
            model_name, model_path = AVAILABLE_MODELS[choice]
            print(f"\nUsing model: {model_name}")
            return load_pickle(model_path), model_name
        else:
            print("Invalid choice, try again.")


def predict(text, model):
    prediction = model.predict([text])[0]

    if prediction == 1:
        label = "SCIENTIFICALLY RELEVANT"
    elif prediction == 4:
        label = "NOT SCIENTIFICALLY RELEVANT"
    else:
        label = f"Unknown label ({prediction})"

    print(f"\nPrediction result: {label}\n")


def main():
    print("=== Scientific Relevance Prediction System ===")

    df = get_dataset()

    text = choose_row(df)

    model, model_name = choose_model()

    predict(text, model)


if __name__ == "__main__":
    main()
