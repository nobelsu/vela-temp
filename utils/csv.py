import json
import pandas as pd
from sklearn.model_selection import train_test_split

def formatRow(row):
    def parse_json_field(field):
        if not field or field.strip() == "":
            return []
        try:
            return json.loads(field)
        except Exception:
            return [] 
    
    industry = row.get("industry", "")
    ipos = parse_json_field(row.get("ipos"))
    acquisitions = parse_json_field(row.get("acquisitions"))
    educations_json = parse_json_field(row.get("educations_json"))
    jobs_json = parse_json_field(row.get("jobs_json"))
    anonymised_prose = row.get("anonymised_prose", "").strip()

    formatted = f"""
        industry: "{industry}",
        ipos: {json.dumps(ipos)},
        acquisitions: {json.dumps(acquisitions)},
        educations_json: {json.dumps(educations_json)},
        jobs_json: {json.dumps(jobs_json)},
        anonymised_prose: \"\"\"
            {anonymised_prose}
        \"\"\"
        """

    return formatted

def split(data_path="data/data.csv", test_path="data/test.csv", train_path="data/train.csv", test_size=80, train_size=320, seed=67, shuffle=True):
    df = pd.read_csv(data_path)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        train_size=train_size,
        random_state=seed,
        shuffle=shuffle
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train set saved to {train_path}, shape: {train_df.shape}")
    print(f"Test set saved to {test_path}, shape: {test_df.shape}")
