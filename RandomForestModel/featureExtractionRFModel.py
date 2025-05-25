import csv
import pandas as pd
import numpy as np
import codecs
import ast
import os
from transformers import AutoTokenizer, AutoModel
import git

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base").cpu()
def codebert_vectorize(code):
    try:
        if isinstance(code, list):
            code = " ".join(code)
        elif not isinstance(code, str):
            code = str(code)      
        inputs = tokenizer(code, return_tensors="pt",truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings
    except Exception as e:
         print(f"Error at line : {e}") 

def get_file_versions(repo,commit_hash, file_path):   
    repo = git.Repo(f"E:/Master thesis/repos/{repo}")        
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
    row_count = 0 
    for row in df.itertuples(index=True):
        print(f"Processing row {row.repo} {row_count+1}")
        row_count += 1
        repo = row.repo
        if(row_count < 1674 and repo=="poetry"): 
            continue

        filepath = row.filepath
        methods = ', '.join(row.parsed_methods)
        include_bugs = row.bug_lines
        path = filepath.replace("\\", "/")
        max_number = max(include_bugs) if len(include_bugs)>0  else 1
        lines = row.split_content
        if isinstance(lines, list) and max_number > len(lines):
            file_content = get_file_versions(repo,row.commit,path)
            try:
                lines = file_content[1].split('\n')
            except AttributeError as e:
                print(e)

        features_data = []
        raw_data=[]
        defect_data=[]       
        if isinstance(lines, list):
            for i, line in enumerate(lines, start=1):
                if line.strip():  
                    label = 1 if i in include_bugs else 0
                    input_text = f"[REPO] {repo} [FILE] {filepath} [METHODS] {methods} [LINE] {i} [CODE] {line}"
                    if label:
                        defect_data.append(input_text)
                    else:
                        raw_data.append(input_text)

                #features_data.append({'input_text': input_text, 'label': label})
            try:
                if(len(defect_data) >0):
                    vec = codebert_vectorize(" ".join(defect_data))            
                    label = np.array([[1]]) 
                    vec_with_label = np.concatenate((vec, label), axis=1)
                    features_data.append(vec_with_label.flatten())  
                if(len(raw_data) >0):
                    vec = codebert_vectorize(" ".join(raw_data))            
                    label = np.array([[0]]) 
                    vec_with_label = np.concatenate((vec, label), axis=1)  
                    features_data.append(vec_with_label.flatten())       
            except Exception as e:
                print(f"Concatenation failed: {e}")

            # log_path = "C:/myenv/myenv/log.csv"
            # log_entry = pd.DataFrame([[repo, len(defect_data), len(raw_data)]],columns=["repo", "defect_count", "raw_count"])

            # if not os.path.exists(log_path):
            #     log_entry.to_csv(log_path, index=False, mode='w', header=True)
            # else:
            #     log_entry.to_csv(log_path+"new", index=False, mode='a', header=False)

            df_f = pd.DataFrame(features_data)        
            if not os.path.exists(f"D:/MSC/SEMESTER3/TextMining/Labs/myenv/fea_{type}/{repo}.csv"):
                df_f.to_csv(f"D:/MSC/SEMESTER3/TextMining/Labs/myenv/fea_{type}/{repo}.csv", quoting=csv.QUOTE_MINIMAL, index=False, mode='w', header=True)
            else:
                df_f.to_csv(f"D:/MSC/SEMESTER3/TextMining/Labs/myenv/fea_{type}/{repo}.csv",  quoting=csv.QUOTE_MINIMAL,index=False, mode='a', header=False)     
    
# repos = ["yolov5","black","jax","redash","pipenv","numpy","openpilot","transformers","localstack","poetry","spaCy","celery","scikit-learn","cpython","airflow","lightning","django","cpython","scikit-learn","celery"]
repos =["spaCy"]
for repo in repos:      
    preprocess_data_line_level(repo, "test")

