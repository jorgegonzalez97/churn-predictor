import mlflow, os

def promote_to_stage(model_name: str, stage: str = "Staging"):
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions(model_name, stages=["None","Staging","Production"])
    if not latest:
        return None
    mv = sorted(latest, key=lambda m: m.version, reverse=True)[0]
    client.transition_model_version_stage(name=model_name, version=mv.version, stage=stage, archive_existing_versions=False)
    return {"name": model_name, "version": mv.version, "stage": stage}

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--stage", default="Staging")
    a = ap.parse_args()
    res = promote_to_stage(a.model_name, a.stage)
    print(json.dumps(res or {}))
