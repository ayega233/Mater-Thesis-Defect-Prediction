from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import numpy as np
import os
import joblib
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import json
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, classification_report


class ModelTrainer:

    def __init__(self):  
        self.train_count = 0

    def train(self,train_repos):
        traind_data = []
        for  re in train_repos:
            train_df = pd.read_csv(f"fea-train/{re}.csv")
            traind_data.append(train_df)
        
        combined_df = pd.concat(traind_data, ignore_index=True)
        print("combined all repository data")
        X_train = combined_df.iloc[:, :-1]
        self.train_count =combined_df.shape[0]
        # X_train = X_train.drop(columns=["num_changes"])
        y_train = combined_df.iloc[:, -1]
        #nan_mask = y_train.isna()

# Get the indices of rows with NaN values
        #nan_indices = y_train[nan_mask].index

# Display the indices
        #print(nan_indices)
              
        param_grid = {
            'n_estimators': [90, 100, 120],
            'max_depth': [4, 5, 6],
            'min_samples_split': [1,2,3]
        }

        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        joblib.dump(self.model, f"random_forest_model_new_model_1.joblib")

    def test(self,repo):

        train_df = pd.read_csv(f"fea_train/{repo}.csv")  
        test_df = pd.read_csv(f"fea_test/{repo}.csv")    
        val_df = pd.read_csv(f"fea_val/{repo}.csv") 
       
        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]

        # X_train = X_train.drop(columns=["num_changes"])
        # y_train = train_df["is_defective"]
    
        X_val = val_df.iloc[:, :-1]
        # X_val = X_val.drop(columns=["num_changes"])
        y_val = val_df.iloc[:, -1]

        X_test = test_df.iloc[:, :-1]
        # X_test = X_test.drop(columns=["num_changes"])
        y_test = test_df.iloc[:, -1]

        y_train_pred = self.model.predict(X_train)

        report_train = classification_report(y_train, y_train_pred, output_dict=True)
        print("Train_Accuracy:", accuracy_score(y_train, y_train_pred))

        y_val_pred = self.model.predict(X_val)

        report_validation = classification_report(y_val, y_val_pred, output_dict=True)
        print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

        
        y_test_pred = self.model.predict(X_test)

        report_test = classification_report(y_test, y_test_pred, output_dict=True)

        print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
        

        results= {
            "repo":repo,
            "training_set":self.train_count,
            "validation_set":val_df.shape[0],
            "test_set":test_df.shape[0],
            "Train_Accuracy": round(accuracy_score(y_train, y_train_pred),4),
            "valiadation_accuracy": round(accuracy_score(y_val, y_val_pred),4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred),4),   
            "train_defect_count":(y_train == 1).sum(),
            "train_non_defect_count":(y_train == 0).sum(),
            "test_defect_count":(y_test == 1).sum(),
            "test_non_defect_count":(y_test == 0).sum(),
            # "validation_defect_count":(y_val == 1).sum(),
            # "validation_non_defect_count":(y_val == 0).sum(),
             **self.best_params
        }

        for class_label, metrics in report_train.items():
            if class_label != 'accuracy':  
                for metric, value in metrics.items():
                    results[f"train_{class_label}_{metric}"] = round(value, 4)

        for class_label, metrics in report_test.items():
            if class_label != 'accuracy':  
                for metric, value in metrics.items():
                    results[f"test_{class_label}_{metric}"] = round(value, 4)

        for class_label, metrics in report_validation.items():
            if class_label != 'accuracy':  
                for metric, value in metrics.items():
                    results[f"validation_{class_label}_{metric}"] = round(value, 4)      
           
        return results
    
def trainIndevidually(repos):
    accuracy_data = []  
    file_path = "accuracy_individual_new_code_pert.csv"
    for repo in repos:                  
        print(f"Training model for {repo}")
        trainer = ModelTrainer()
        trainer.train([repo])
        accuracy_data = []  
        accuracy_data.append(trainer.test(repo))
        df_f = pd.DataFrame(accuracy_data)
        if not os.path.exists(file_path):
            df_f.to_csv(file_path, index=False, mode='w', header=True)
        else:
            df_f.to_csv(file_path, index=False, mode='a', header=False)
       

def trainAll(repos):
    accuracy_data = []  
    trainer = ModelTrainer()
    trainer.train(repos)
    #file_path = "accuracy_trainall_new_model_1.csv"
    #for repo in repos:                  
    #    print(f"Testing model for {repo}")     
     #   accuracy_data = []    
     #   accuracy_data.append(trainer.test(repo))
    #    df_f = pd.DataFrame(accuracy_data)
    ##    if not os.path.exists(file_path):
    #        df_f.to_csv(file_path, index=False, mode='w', header=True)
    #    else:
    #        df_f.to_csv(file_path, index=False, mode='a', header=False)

def trainSomeTestOther(trainRepo,testRepo):
    accuracy_data = []  
    trainer = ModelTrainer()
    trainer.train(trainRepo)
    for repo in testRepo:                  
        print(f"Testing model for {repo}")        
        accuracy_data.append(trainer.test(repo))
    return accuracy_data

def test_all(repos):  
    print("Testing all")
    loaded_model = joblib.load('random_forest_model_new_model_1.joblib')

    train_data = []
    print("Model loaded")
    for  re in repos:
        train_df = pd.read_csv(f"fea-train/{re}.csv")
        train_data.append(train_df)
    
    combined_df_train = pd.concat(train_data, ignore_index=True)
    print("Data combined")
    X_train = combined_df_train.iloc[:, :-1]
    y_train = combined_df_train.iloc[:, -1]

    y_train_pred = loaded_model.predict(X_train)
    # AUC Score
    aoc_train = roc_auc_score(y_train, y_train_pred)
    # results[split] = {"auc": aoc}
    print("AUC Score:", aoc_train)
    # ROC Curve
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    report_train = classification_report(y_train, y_train_pred, output_dict=True)

    test_data = []
    print("Model loaded")
    for  re in repos:
        train_df = pd.read_csv(f"fea_test/{re}.csv")
        test_data.append(train_df)
    
    combined_df_test = pd.concat(test_data, ignore_index=True)
    print("Data combined")
    X_test = combined_df_test.iloc[:, :-1]
    y_test = combined_df_test.iloc[:, -1]

    y_test_pred = loaded_model.predict(X_test)
    # AUC Score
    aoc_test = roc_auc_score(y_test, y_test_pred)
    # results[split] = {"auc": aoc}
    print("AUC Score:", aoc_test)
    # ROC Curve
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
    report_test = classification_report(y_test, y_test_pred, output_dict=True)

    val_data = []
    for  re in repos:
        train_df = pd.read_csv(f"fea_val/{re}.csv")
        val_data.append(train_df)
    
    combined_df_val = pd.concat(val_data, ignore_index=True)
    print("Data combined val")
    X_val = combined_df_val.iloc[:, :-1]
    y_val = combined_df_val.iloc[:, -1]
    
    y_val_pred = loaded_model.predict(X_val)
    # AUC Score
    aoc_val = roc_auc_score(y_val, y_val_pred)
    # results[split] = {"auc": aoc}
    print("AUC Score: Val", aoc_val)
    # ROC Curve
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
    report_validation = classification_report(y_val, y_val_pred, output_dict=True)
    results= {
        "train_set":combined_df_train.shape[0],
        "validation_set":combined_df_val.shape[0],
        "test_set":combined_df_test.shape[0],
        "Train_Accuracy": round(accuracy_score(y_train, y_train_pred),4),
        "valiadation_accuracy": round(accuracy_score(y_val, y_val_pred),4),
        "test_accuracy": round(accuracy_score(y_test, y_test_pred),4),   
        "train_defect_count":(y_train == 1).sum(),
        "train_non_defect_count":(y_train == 0).sum(),
        "test_defect_count":(y_test == 1).sum(),
        "test_non_defect_count":(y_test == 0).sum(),    
        "precision_train" : round(precision_score(y_train, y_train_pred),4),
        "recall_train":  round(recall_score(y_train, y_train_pred),4),
        "auc_train" :  round(roc_auc_score(y_train, y_train_pred),4),
        "precision_test" : round(precision_score(y_test, y_test_pred),4),
        "recall_test":  round(recall_score(y_test, y_test_pred),4),
        "auc_test" :  round(roc_auc_score(y_test, y_test_pred),4),
        "precision_val" : round(precision_score(y_val, y_val_pred),4),
        "recall_val":  round(recall_score(y_val, y_val_pred),4),
        "auc_val" :  round(roc_auc_score(y_val, y_val_pred),4),   
    }

    for class_label, metrics in report_train.items():
        if class_label != 'accuracy':  
            for metric, value in metrics.items():
                results[f"train_{class_label}_{metric}"] = round(value, 4)

        for class_label, metrics in report_test.items():
            if class_label != 'accuracy':  
                for metric, value in metrics.items():
                    results[f"test_{class_label}_{metric}"] = round(value, 4)

        for class_label, metrics in report_validation.items():
            if class_label != 'accuracy':  
                for metric, value in metrics.items():
                    results[f"validation_{class_label}_{metric}"] = round(value, 4) 

    accuracy_data = []  
    accuracy_data.append(results)
    df_f = pd.DataFrame(accuracy_data)
    file_path = "accuracy_new_model_all_1234.csv"
    if not os.path.exists(file_path):
        df_f.to_csv(file_path, index=False, mode='w', header=True)
    else:
        df_f.to_csv(file_path, index=False, mode='a', header=False)
    # Plot and save ROC curve
    plt.figure()
    plt.plot(fpr_train, tpr_train, label=f'AUC = {aoc_train:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Train')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(fpr_test, tpr_test, label=f'AUC = {aoc_test:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Test')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    plt.figure()
    plt.plot(fpr_val, tpr_val, label=f'AUC = {aoc_val:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Validation')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #conf_matrix_train = confusion_matrix(y_test, y_test_pred) 

def print_confusion_metrix(repos):  
    loaded_model = joblib.load('random_forest_model_all.joblib')
    for repo in repos:                  
                 
        train_df = pd.read_csv(f"feature_train/{repo}.csv")  
        test_df = pd.read_csv(f"feature_test/{repo}.csv")    
        val_df = pd.read_csv(f"feature_validation/{repo}.csv") 
       
        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]

        X_val = val_df.iloc[:, :-1]       
        y_val = val_df.iloc[:, -1]

        X_test = test_df.iloc[:, :-1]        
        y_test = test_df.iloc[:, -1]

        y_train_pred = loaded_model.predict(X_train)
        conf_matrix_train = confusion_matrix(y_train, y_train_pred) 
        y_val_pred = loaded_model.predict(X_val)
        conf_matrix_val = confusion_matrix(y_val, y_val_pred) 

        y_test_pred = loaded_model.predict(X_test)
        conf_matrix_test = confusion_matrix(y_test, y_test_pred) 
        
        print(repo)
        print(conf_matrix_train)
        print(conf_matrix_val)
        print(conf_matrix_test)

def get_all_data(repos,data_type):
    train_data = {}
    for repo in repos:    
        train_df = pd.read_csv(f"feature_{data_type}/{repo}.csv")
        train_data[repo]=train_df
    combined_df = pd.concat(train_data, ignore_index=True)
    return combined_df
       

  
def predict_on_all(repos):  
    loaded_model = joblib.load('random_forest_model_all.joblib')            
    train_df = get_all_data(repos,"train")
    test_df =get_all_data(repos,"test") 
    val_df = get_all_data(repos,"validation")
    
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    X_val = val_df.iloc[:, :-1]       
    y_val = val_df.iloc[:, -1]

    X_test = test_df.iloc[:, :-1]        
    y_test = test_df.iloc[:, -1]

    y_train_pred = loaded_model.predict(X_train)

    report_train = classification_report(y_train, y_train_pred, output_dict=True)
    print("Train_Accuracy:", accuracy_score(y_train, y_train_pred))

    y_val_pred = loaded_model.predict(X_val)

    report_validation = classification_report(y_val, y_val_pred, output_dict=True)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

    
    y_test_pred = loaded_model.predict(X_test)

    report_test = classification_report(y_test, y_test_pred, output_dict=True)

    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    

    results= {        
        "validation_set":val_df.shape[0],
        "test_set":test_df.shape[0],
        "Train_Accuracy": round(accuracy_score(y_train, y_train_pred),4),
        "valiadation_accuracy": round(accuracy_score(y_val, y_val_pred),4),
        "test_accuracy": round(accuracy_score(y_test, y_test_pred),4),   
        "train_defect_count":(y_train == 1).sum(),
        "train_non_defect_count":(y_train == 0).sum(),
        "test_defect_count":(y_test == 1).sum(),
        "test_non_defect_count":(y_test == 0).sum(),       
    }

    for class_label, metrics in report_train.items():
        if class_label != 'accuracy':  
            for metric, value in metrics.items():
                results[f"train_{class_label}_{metric}"] = round(value, 4)

    for class_label, metrics in report_test.items():
        if class_label != 'accuracy':  
            for metric, value in metrics.items():
                results[f"test_{class_label}_{metric}"] = round(value, 4)

    for class_label, metrics in report_validation.items():
        if class_label != 'accuracy':  
            for metric, value in metrics.items():
                results[f"validation_{class_label}_{metric}"] = round(value, 4)      
        
    
    file_path = "accuracy_trainall_test_all_111.csv"
    print(results)
    accuracy_data = []  
    accuracy_data.append(results)
    df_f = pd.DataFrame(accuracy_data)
    if not os.path.exists(file_path):
        df_f.to_csv(file_path, index=False, mode='w', header=True)
    else:
        df_f.to_csv(file_path, index=False, mode='a', header=False)


def getStat(repos):
    statistic_data = []  
   
    for repo in repos:                  
        train_df = pd.read_csv(f"train/{repo}_feature.csv")  
        test_df = pd.read_csv(f"test/{repo}_feature.csv")    
        # val_df = pd.read_csv(f"val/{repo}_feature.csv") 
        print(train_df[train_df['is_defective'] == False].head())
        statistic_data.append({
            "repo":repo,
            "train_defect_count":len(train_df[train_df['is_defective'] == True]),
            "train_non_defect_count":len(train_df[train_df['is_defective'] == False]),
            "test_defect_count":len(test_df[test_df['is_defective'] == True]),
            "test_non_defect_count":len(test_df[test_df['is_defective'] ==False]),
            # "validation_defect_count":len(val_df[val_df['is_defective'] == True]),
            # "validation_non_defect_count":len(val_df[val_df['is_defective'] == False]),
        })
    df_f = pd.DataFrame(statistic_data)
    df_f.to_csv("statistics_repo_test.csv", index=False) 

if __name__ == "__main__":
    #,"django"
    #repos = ["airflow","cpython","scikit-learn","celery","transformers","localstack","spaCy","yolov5","numpy","jax","poetry","openpilot","black","lightning","pandas","sentry","django","ray","redash","scrapy","pipenv"]
   #trainer = ModelTrainer("scikit-learn")
    #trainer = ModelTrainer("cpython")
    #trainer.train()
    #repos = ["pipenv","jax"]
   
    repos = [
    "airflow", "ansible", "black", "celery", "core", "cpython", "django", "jax",
    "lightning", "localstack", "numpy", "openpilot", "pandas", "pipenv", "poetry",
    "ray", "redash", "scikit-learn", "sentry","scrapy", "spaCy", "transformers", "yolov5"
    ]
    #repos = ["ansible"]
    #trainIndevidually(repos)
   # trainAll(repos)
    test_all(repos)
   # print_confusion_metrix(repos)
    #predict_on_all(repos)
    #getStat(repos)
    # trainer = ModelTrainer()
    #trainer.train(["airflow","cpython","scikit-learn","celery"])
    # repo_list = np.array(repos)

    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # accuracy_data = []  
    # for train_idx, test_idx in kf.split(repo_list):
    #     print(train_idx,test_idx)
    #     train_repos = repo_list[np.array(train_idx)]
    #     test_repos = repo_list[np.array(test_idx)]
    #     accuracy_data.append(trainSomeTestOther(train_repos,test_repos))

    # df_f = pd.DataFrame(accuracy_data)
    # df_f.to_csv(f"accuracy_train_some_test_other.csv", index=False) 