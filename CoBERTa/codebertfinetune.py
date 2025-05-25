import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

# Load the dataset
# df = pd.read_csv("your_dataset.csv")  # Replace with your actual file

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Define the PyTorch dataset
class CodeDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Function to fine-tune CodeBERT for a specific repository
def fine_tune_codebert(repo_name, repo_dft, repo_dfv):
    train_texts = repo_dft.iloc[:, 0].astype(str).tolist()
    train_labels = repo_dft.iloc[:, 1].tolist()

    val_texts= repo_dft.iloc[:, 0].astype(str).tolist()
    val_labels= repo_dfv.iloc[:, 1].tolist()

    train_dataset = CodeDataset(train_texts, train_labels)
    val_dataset = CodeDataset(val_texts, val_labels)

    model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

    training_args = TrainingArguments(
            output_dir=f"./results_{repo_name}",
            evaluation_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_ratio=0.1,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir=f"./logs_{repo_name}",
            logging_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    results_df = pd.DataFrame([eval_results])

    # Save the results to a CSV file
    results_df.to_csv(f"./results_{repo_name}/evaluation_metrics.csv", index=False)

    model.save_pretrained(f"./codebert_defect_model_{repo_name}")
    tokenizer.save_pretrained(f"./codebert_defect_model_{repo_name}")

    print(f"Training complete for repository {repo_name}. Model saved to './codebert_defect_model_{repo_name}'.")

# Fine-tune CodeBERT for each repository

# repos = ["pipenv""yolov5","black","jax","redash","pipenv","numpy","openpilot","transformers","localstack","poetry","spaCy","celery","scikit-learn","cpython","airflow","lightning",]
# repos =["django","pandas","ray","core","ansible","sentry","scrapy"]
repos =["pipenv"]
for repo in repos:      
    repo_df_train =pd.read_csv(f"D:/MSC/SEMESTER3/TextMining/Labs/myenv/fea_cordbertpr_train/{repo}.csv")
    repo_df_val =pd.read_csv(f"D:/MSC/SEMESTER3/TextMining/Labs/myenv/fea_cordbertpr_val/{repo}.csv")
    fine_tune_codebert(repo,repo_df_train,repo_df_val)

# Fine-tune CodeBERT on the aggregated dataset
# fine_tune_codebert("aggregated", df)
