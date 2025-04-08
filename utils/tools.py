import os
import pandas as pd

def get_latest_model_path():
    models_dir = "./models"
    if not os.path.exists(models_dir):
        return None
    models = [f for f in os.listdir(models_dir) if f.startswith("face_model") and f.endswith(".yml")]
    if not models:
        return None
    return os.path.join(models_dir, sorted(models)[-1])

def load_names():
    names_path = "./data/names.csv"
    names = []
    if os.path.exists(names_path):
        df = pd.read_csv(names_path)
        names = df['name'].tolist()
    return names
