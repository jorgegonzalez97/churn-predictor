import pandas as pd
import yaml
from pathlib import Path

def load_config():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def prepare_dataset(input_csv: str, output_train_csv: str, output_test_csv: str) -> None:
    cfg = load_config()
    df = pd.read_csv(input_csv)
    if 'customerID' in df.columns and 'customer_id' not in df.columns:
        df = df.rename(columns={'customerID': 'customer_id'})
    # Simple cleaning
    df = df.dropna(axis=0).reset_index(drop=True)
    target = cfg.get("target_column", "churn")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=cfg.get("test_size", 0.2), random_state=cfg.get("random_state", 42), stratify=df[target])
    Path(output_train_csv).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_train_csv, index=False)
    test.to_csv(output_test_csv, index=False)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train_out", required=True)
    ap.add_argument("--test_out", required=True)
    args = ap.parse_args()
    prepare_dataset(args.input, args.train_out, args.test_out)
