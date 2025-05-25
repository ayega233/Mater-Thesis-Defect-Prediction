import csv
import pandas as pd
import numpy as np
import codecs
import ast
import os
from transformers import AutoTokenizer, AutoModel
import git

def get_file_versions(repo,commit_hash, file_path):   
    repo = git.Repo(f"D:/Master thesis/repos/{repo}")        
    commit = repo.commit(commit_hash)
    
    parent_commit = commit.parents[0] if commit.parents else None
    
    if parent_commit:
        
        current_version = None            
        try:
            current_version = repo.git.show(f"{commit_hash}:{file_path}")
        except git.exc.GitCommandError:
            current_version = None 
            
        previous_version = None           
        try:
            previous_version = repo.git.show(f"{parent_commit.hexsha}:{file_path}")
        except BaseException  as e:
                print(f"error has occured for file_path {file_path} {e}")
        
        return previous_version, current_version
    else:
        raise ValueError("The given commit does not have a parent commit .")

def preprocess_data_line_level(repo,type):
    print(f"reading file {repo} ")
    df = pd.read_excel(f"D:/MSC/SEMESTER3/TextMining/Labs/myenv/{type}/{type}_{repo}.xlsx")

    df['parsed_methods'] = df['methods'].str.strip('[]').str.split(', ')
    df['bug_lines'] = df['lines'].str.strip('[]').apply(
        lambda x: np.fromstring(x, sep=' ', dtype=int)
    )
    df['split_content'] = df['content'].str.replace('\\n', '\n').str.split('\n')

    features_data = []         
    for row in df.itertuples(index=True):
        print(f"Processing row {row.repo} {row.index}")
        repo = row.repo
        filepath = row.filepath
        methods = ', '.join(row.parsed_methods)
        include_bugs = row.bug_lines
        path = filepath.replace("\\", "/")
        file_content = get_file_versions(repo,row.commit,path)
        try:
            lines = file_content[1].split('\n')
        except AttributeError as e:
            print(e)

        features_data = []
        raw_data=[]
        defect_data=[]  
        print(len(lines))
        for i, line in enumerate(lines, start=1):
            if line.strip():  
                label = 1 if i in include_bugs else 0
                if label:
                    defect_data.append(line)
                else:
                    raw_data.append(line)

        if defect_data:
            joined_defect_code = " ".join(defect_data)
            input_text = f"[REPO] {repo} [FILE] {filepath} [METHODS] {methods} [CODE]{joined_defect_code}"
            features_data.append([input_text, 1])

        if raw_data:
            joined_raw_code = " ".join(raw_data)
            input_text = f"[REPO] {repo} [FILE] {filepath} [METHODS] {methods} [CODE]{joined_raw_code}"
            features_data.append([input_text, 0])


        df_f = pd.DataFrame(features_data)      
        df_f = df_f.applymap(lambda x: x.encode('utf-8', 'replace').decode('utf-8') if isinstance(x, str) else x)  
        if not os.path.exists(f"D:/MSC/SEMESTER3/TextMining/Labs/myenv/fea_cordbertpr_{type}/{repo}.csv"):
            df_f.to_csv(f"D:/MSC/SEMESTER3/TextMining/Labs/myenv/fea_cordbertpr_{type}/{repo}.csv", quoting=csv.QUOTE_MINIMAL, index=False, mode='w', header=True,encoding='utf-8')
        else:
            df_f.to_csv(f"D:/MSC/SEMESTER3/TextMining/Labs/myenv/fea_cordbertpr_{type}/{repo}.csv",  quoting=csv.QUOTE_MINIMAL,index=False, mode='a', header=False,encoding='utf-8')     

# "yolov5","black","jax","redash","pipenv", 
# repos =["numpy","openpilot","transformers","localstack","poetry","spaCy","celery","scikit-learn","cpython","airflow","lightning","django","pandas","ray","core","ansible","sentry","scrapy"]
repos =["airflow","lightning","django","pandas","ray","core","ansible","sentry","scrapy"]
for repo in repos:      
    preprocess_data_line_level(repo, "train")