import json
import pandas as pd

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    df['input_text'] = 'generate question: ' + df['answer'] + ' </s> difficulty: ' + df['difficulty']
    df['target_text'] = df['followup_question']
    return df.reset_index(drop=True)
