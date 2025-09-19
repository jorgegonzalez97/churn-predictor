import pandas as pd, json
from evidently.report import Report
from evidently.metrics import DataDriftPreset

def drift_report(ref_csv: str, cur_csv: str, out_html: str):
    ref = pd.read_csv(ref_csv)
    cur = pd.read_csv(cur_csv)
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=ref, current_data=cur)
    rep.save_html(out_html)
    return {"report_path": out_html}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_csv", required=True)
    ap.add_argument("--cur_csv", required=True)
    ap.add_argument("--out_html", required=True)
    a = ap.parse_args()
    print(json.dumps(drift_report(a.ref_csv, a.cur_csv, a.out_html)))
