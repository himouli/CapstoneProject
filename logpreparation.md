Here's a step-by-step guide to building a full pipeline to prepare OpenStack logs (from LogHub) for LogBERT, including parsing, cleaning, preprocessing, and getting ready for training or fine-tuning.

🧱 Step-by-Step Pipeline: OpenStack Logs → LogBERT Input
✅ Step 1: Download and Inspect the Dataset
bash
Copy
Edit
git clone https://github.com/logpai/loghub.git
cd loghub/OpenStack
You’ll find:

OpenStack_2k.log: raw logs

OpenStack_2k_label.csv: anomaly labels (optional for evaluation)

✅ Step 2: Install Required Libraries
bash
Copy
Edit
pip install drain3 pandas nltk scikit-learn transformers
✅ Step 3: Parse Logs Using Drain3
Drain is a structured log parser that turns raw logs into templates.

python
Copy
Edit
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
import json

log_file = "OpenStack_2k.log"
persistence = FilePersistence("drain3_state.json")
template_miner = TemplateMiner(persistence)

parsed_logs = []
with open(log_file, 'r') as f:
    for line in f:
        result = template_miner.add_log_message(line.strip())
        if result["change_type"] != "none":
            print(f"Template extracted: {result['template_mined']}")
        parsed_logs.append(result['template_mined'])
Each template_mined will be a string like:

arduino
Copy
Edit
"Instance {instance_id} failed to spawn"
✅ Step 4: Build Sequences from Parsed Templates
LogBERT expects sequences of templates (like sentences in NLP).

python
Copy
Edit
import numpy as np

# Create sliding windows of templates
window_size = 10
sequences = []

for i in range(len(parsed_logs) - window_size):
    sequence = parsed_logs[i:i + window_size]
    sequences.append(sequence)
✅ Step 5: Tokenization and Preparation for BERT
python
Copy
Edit
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_sequence(seq):
    joined = " [SEP] ".join(seq)
    return tokenizer(joined, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

tokenized_inputs = [tokenize_sequence(seq) for seq in sequences]
✅ Step 6: Optional – Prepare Labels for Supervised Fine-Tuning
python
Copy
Edit
import pandas as pd

label_file = "OpenStack_2k_label.csv"
labels_df = pd.read_csv(label_file)
labels = labels_df['label'].values  # 0 = Normal, 1 = Anomaly

# Align labels with sequences (if log lines and templates are 1:1)
sequence_labels = []
for i in range(len(labels) - window_size):
    sequence_labels.append(int(1 in labels[i:i + window_size]))  # Label is 1 if any log in the sequence is anomalous
✅ Step 7: Prepare Dataset for Model
python
Copy
Edit
from torch.utils.data import Dataset

class LogDataset(Dataset):
    def __init__(self, tokenized_inputs, labels):
        self.inputs = tokenized_inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = self.inputs[idx]
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
✅ Step 8: Model Training or Fine-Tuning
You can now fine-tune LogBERT or BERT with a classification head on your LogDataset.

python
Copy
Edit
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./logbert_openstack",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
✅ Step 9: Visualize Anomaly Scores
During evaluation, output anomaly scores per sequence, and plot them over time.

python
Copy
Edit
import matplotlib.pyplot as plt

scores = trainer.predict(eval_dataset).predictions[:, 1]  # Probability of class 1 (anomaly)
plt.plot(scores)
plt.title("Anomaly Scores Over Time")
plt.xlabel("Sequence Index")
plt.ylabel("Anomaly Score")
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()
✅ Summary of Pipeline
Step	Description
Parse Logs	Use Drain to convert raw logs into templates
Create Sequences	Sliding window of log templates (size ~10)
Tokenize & Prepare Input	Convert sequences to BERT-compatible input
Label Sequences	Optional step using LogHub labels
Train/Fine-Tune	Train BERT-based classifier (LogBERT-like)
Visualize Output	Plot anomaly scores over time

Let me know if you'd like a Colab notebook version of this pipeline or want to extend this for streaming logs from OpenStack in near real-time.







