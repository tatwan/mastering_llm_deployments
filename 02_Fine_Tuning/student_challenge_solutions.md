# Student Challenge Solutions Guide

**Module 02: Fine-Tuning Transformers**

This guide provides complete solutions and detailed explanations for all student challenges in notebooks 01-03. Use this to check your work or if you get stuck.

---

## 📘 Challenge 1: AG News Classification (Notebook 01)

**Objective**: Fine-tune DistilBERT for 4-class text classification using the AG News dataset.

### 🎯 What You'll Learn

- How to adapt a binary classifier to multi-class classification
- Working with different dataset structures
- Mapping custom labels to model configuration

### 💡 Key Insights

| Concept | Binary (Rotten Tomatoes) | Multi-class (AG News) |
|---------|--------------------------|----------------------|
| `num_labels` | 2 | 4 |
| Labels | Positive/Negative | World, Sports, Business, Sci/Tech |
| Dataset column | `text` | `text` |
| Output layer | 768 → 2 | 768 → 4 |

> **Why does this work?** The pre-trained model already understands language structure. We only need to replace the final layer (classification head) to output 4 classes instead of 2.

---

### ✅ Complete Solution

```python
# =============================================================
# CHALLENGE 1: Fine-tune on AG News (4-class classification)
# =============================================================

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate
import numpy as np

# 1. Load the AG News dataset
print("Loading AG News dataset...")
dataset_ag = load_dataset("ag_news")

print(f"Dataset structure: {dataset_ag}")
print(f"\nTrain examples: {len(dataset_ag['train']):,}")
print(f"Test examples: {len(dataset_ag['test']):,}")

# Explore the data
print("\nSample examples:")
for i in range(3):
    example = dataset_ag['train'][i]
    print(f"  Label {example['label']}: {example['text'][:80]}...")
```

```python
# 2. Define label mappings
# AG News has 4 categories: World (0), Sports (1), Business (2), Sci/Tech (3)
labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
id2label = {k: v for k, v in labels.items()}
label2id = {v: k for k, v in labels.items()}

print("Label mappings:")
for id, label in id2label.items():
    print(f"  {id} → {label}")
```

```python
# 3. Prepare tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding=True,
        truncation=True,
        max_length=256
    )

# 4. Create smaller subsets for faster training (optional - use full for better results)
train_size = 5000  # Use more for better accuracy
test_size = 1000

train_data = dataset_ag['train'].shuffle(seed=42).select(range(train_size))
test_data = dataset_ag['test'].shuffle(seed=42).select(range(test_size))

# Create validation split
train_val_split = train_data.train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

# Tokenize
train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)
test_tokenized = test_data.map(tokenize_function, batched=True)

print(f"Training samples: {len(train_tokenized)}")
print(f"Validation samples: {len(val_tokenized)}")
print(f"Test samples: {len(test_tokenized)}")
```

```python
# 5. Initialize model with 4 labels (KEY DIFFERENCE!)
model_ag = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,        # ← Changed from 2 to 4
    id2label=id2label,
    label2id=label2id
)

total_params = sum(p.numel() for p in model_ag.parameters())
print(f"Model parameters: {total_params:,}")
```

```python
# 6. Setup training
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./ag_news_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model_ag,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
```

```python
# 7. Train!
print("Starting training...")
trainer.train()
print("\nTraining complete!")

# 8. Evaluate on test set
test_results = trainer.evaluate(test_tokenized)
print(f"\nTest Accuracy: {test_results['eval_accuracy']:.2%}")
```

```python
# 9. Test on custom examples
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model=model_ag,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

test_texts = [
    "The stock market reached new highs today as investors showed confidence.",
    "Manchester United defeated Liverpool in a thrilling match.",
    "NASA launched a new satellite to study climate change.",
    "World leaders met at the UN to discuss the ongoing crisis."
]

print("Predictions:")
print("=" * 60)
for text in test_texts:
    result = classifier(text)[0]
    print(f"Text: {text[:50]}...")
    print(f"  → {result['label']} ({result['score']:.2%})\n")
```

---

## 📗 Challenge 2: Learning Rate Experiment (Notebook 02)

**Objective**: Run a systematic experiment to see how learning rate affects model performance.

### 🎯 What You'll Learn

- The impact of learning rate on fine-tuning
- How to run hyperparameter experiments
- Interpreting F1 scores across different settings

### 💡 Key Insights

| Learning Rate | Expected Behavior |
|--------------|-------------------|
| 1e-5 (too small) | Slow learning, may underfit |
| 2e-5 to 5e-5 (optimal) | Best balance for BERT models |
| 1e-4 (too large) | Unstable training, may overfit or diverge |

> **Why does learning rate matter so much?** Pre-trained models have already learned useful features. A large learning rate can "unlearn" these features (catastrophic forgetting), while a tiny rate won't adapt the model to your task.

---

### ✅ Complete Solution

```python
# =============================================================
# CHALLENGE 2: Learning Rate Experimentation
# =============================================================

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate
import numpy as np

# Load and prepare data (same as the main notebook)
dataset = load_dataset("imdb")
train_size = 1000  # Reduced for faster experiments
test_size = 200

train_data = dataset['train'].shuffle(seed=42).select(range(train_size))
val_data = dataset['test'].shuffle(seed=42).select(range(test_size))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding=True,
        truncation=True,
        max_length=256
    )

train_tokenized = train_data.map(tokenize_function, batched=True)
val_tokenized = val_data.map(tokenize_function, batched=True)

# Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy.compute(predictions=preds, references=labels)['accuracy'],
        'f1': f1.compute(predictions=preds, references=labels)['f1']
    }

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

```python
# The Experiment Loop
learning_rates = [1e-5, 2e-5, 5e-5, 1e-4]  # From too small to too large
results = []

print("Learning Rate Experiment")
print("=" * 50)

for lr in learning_rates:
    print(f"\n🔄 Training with LR: {lr}...")
    
    # IMPORTANT: Re-initialize model for each experiment!
    # Otherwise, you're continuing from the previous model's weights.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    # Update training args with new learning rate
    training_args = TrainingArguments(
        output_dir=f"./experiment_lr_{lr}",
        num_train_epochs=2,  # Reduced for speed
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=lr,  # ← The variable we're testing!
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",  # Don't save intermediate models
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    
    results.append({
        'lr': lr,
        'accuracy': metrics['eval_accuracy'],
        'f1': metrics['eval_f1']
    })
    
    print(f"  → Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"  → F1 Score: {metrics['eval_f1']:.4f}")
```

```python
# Visualize Results
import matplotlib.pyplot as plt

print("\n" + "=" * 50)
print("EXPERIMENT RESULTS SUMMARY")
print("=" * 50)

lrs = [r['lr'] for r in results]
f1_scores = [r['f1'] for r in results]
accuracies = [r['accuracy'] for r in results]

# Print table
print(f"\n{'Learning Rate':<15} {'Accuracy':<12} {'F1 Score':<12}")
print("-" * 40)
for r in results:
    print(f"{r['lr']:<15} {r['accuracy']:<12.4f} {r['f1']:<12.4f}")

# Find best
best_idx = np.argmax(f1_scores)
print(f"\n🏆 Best Learning Rate: {lrs[best_idx]} (F1: {f1_scores[best_idx]:.4f})")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{lr:.0e}" for lr in lrs]
x = range(len(lrs))

ax.bar([i - 0.2 for i in x], accuracies, width=0.4, label='Accuracy', color='steelblue')
ax.bar([i + 0.2 for i in x], f1_scores, width=0.4, label='F1 Score', color='coral')

ax.set_xlabel('Learning Rate')
ax.set_ylabel('Score')
ax.set_title('Learning Rate vs Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

### 📊 Expected Results Interpretation

| Learning Rate | Typical Outcome |
|--------------|-----------------|
| **1e-5** | F1 ~0.75-0.80 — Model learns slowly, may need more epochs |
| **2e-5** | F1 ~0.82-0.86 — **Sweet spot** for BERT-style models |
| **5e-5** | F1 ~0.80-0.85 — Good, but watch for overfitting |
| **1e-4** | F1 ~0.50-0.70 — **Too high!** Training may be unstable |

> ⚠️ **Note**: Results will vary based on your random seed and data subset. The key insight is the relative pattern, not exact numbers.

---

## 📙 Challenge 3: CNN/DailyMail Summarization (Notebook 03)

**Objective**: Adapt the DialogSum summarization code to work with news articles from CNN/DailyMail.

### 🎯 What You'll Learn

- How to adapt preprocessing for different dataset structures
- Understanding dataset column name differences
- Applying the same model to a different domain

### 💡 Key Insights

| Aspect | DialogSum | CNN/DailyMail |
|--------|-----------|---------------|
| Input column | `dialogue` | `article` |
| Target column | `summary` | `highlights` |
| Content type | Conversations | News articles |
| Typical length | ~100-200 words | ~500-1000 words |

> **What changes?** Only the preprocessing function! The model architecture, training loop, and evaluation stay the same.

---

### ✅ Complete Solution

```python
# =============================================================
# CHALLENGE 3: Fine-tune on CNN/DailyMail News Articles
# =============================================================

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import evaluate
import numpy as np
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# 1. Load CNN/DailyMail dataset
# NOTE: This dataset is large! We use a small slice for the challenge.
print("Loading CNN/DailyMail dataset (small subset)...")
cnn_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:500]")

# Inspect the structure
print(f"\nDataset columns: {cnn_dataset.column_names}")
print(f"Number of examples: {len(cnn_dataset)}")

# Look at a sample
print("\n" + "=" * 60)
print("SAMPLE ARTICLE:")
print("=" * 60)
print(cnn_dataset[0]['article'][:500] + "...")
print("\nHIGHLIGHTS (Summary):")
print(cnn_dataset[0]['highlights'])
```

```python
# 2. Load model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# T5 uses a prefix for the task
prefix = "summarize: "

# Parameters
max_input_length = 512
max_target_length = 128

print(f"\nModel: {model_name}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

```python
# 3. Create NEW preprocessing function for CNN/DailyMail
# KEY DIFFERENCE: Column names are 'article' and 'highlights' (not 'dialogue' and 'summary')

def preprocess_news(examples):
    """Tokenize news articles and their highlights for seq2seq training."""
    
    # Use 'article' instead of 'dialogue'
    inputs = [prefix + doc for doc in examples['article']]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding=True
    )
    
    # Use 'highlights' instead of 'summary'
    labels = tokenizer(
        examples['highlights'],
        max_length=max_target_length,
        truncation=True,
        padding=True
    )
    
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs
```

```python
# 4. Split and tokenize
# Create train/validation split
train_val_split = cnn_dataset.train_test_split(test_size=0.1, seed=42)
train_data = train_val_split['train']
val_data = train_val_split['test']

# Apply preprocessing
train_tokenized = train_data.map(preprocess_news, batched=True)
val_tokenized = val_data.map(preprocess_news, batched=True)

print(f"Training samples: {len(train_tokenized)}")
print(f"Validation samples: {len(val_tokenized)}")
```

```python
# 5. Setup ROUGE metrics (same as DialogSum)
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vocab_size = tokenizer.vocab_size
    
    # Clean predictions
    predictions_clean = []
    for row in predictions:
        clean_row = []
        for tok in row:
            tok_int = int(tok)
            if tok_int < 0 or tok_int >= vocab_size:
                clean_row.append(pad_id)
            else:
                clean_row.append(tok_int)
        predictions_clean.append(clean_row)
    
    decoded_preds = tokenizer.batch_decode(predictions_clean, skip_special_tokens=True)
    
    # Clean labels
    labels_clean = []
    for row in labels:
        clean_row = []
        for tok in row:
            tok_int = int(tok)
            if tok_int < 0 or tok_int >= vocab_size:
                clean_row.append(pad_id)
            else:
                clean_row.append(tok_int)
        labels_clean.append(clean_row)
    
    decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
    
    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    return {
        'rouge1': result['rouge1'],
        'rouge2': result['rouge2'],
        'rougeL': result['rougeL']
    }
```

```python
# 6. Setup training
training_args = Seq2SeqTrainingArguments(
    output_dir="./cnn_summarization_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Smaller batch size (news articles are longer)
    per_device_eval_batch_size=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    predict_with_generate=True,
    generation_max_length=max_target_length,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    logging_steps=25,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
```

```python
# 7. Train!
print("Starting training...")
print("=" * 50)
trainer.train()
print("\nTraining complete!")

# 8. Evaluate
eval_results = trainer.evaluate()
print("\nEvaluation Results:")
print("=" * 40)
print(f"ROUGE-1: {eval_results['eval_rouge1']:.2%}")
print(f"ROUGE-2: {eval_results['eval_rouge2']:.2%}")
print(f"ROUGE-L: {eval_results['eval_rougeL']:.2%}")
```

```python
# 9. Test on examples
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Test on a validation example
test_article = val_data[0]['article']
reference = val_data[0]['highlights']

generated = summarizer(
    prefix + test_article,
    max_length=128,
    min_length=30,
    do_sample=False
)[0]['summary_text']

print("\n" + "=" * 60)
print("GENERATED SUMMARY TEST")
print("=" * 60)
print("\nARTICLE (truncated):")
print(test_article[:500] + "...")
print("\nREFERENCE HIGHLIGHTS:")
print(reference)
print("\nGENERATED SUMMARY:")
print(generated)
```

---

## 🔧 Troubleshooting Common Issues

### Out of Memory (OOM) Errors

```python
# Reduce batch size
per_device_train_batch_size=4  # or even 2

# Reduce sequence length
max_input_length=256  # instead of 512

# Use gradient accumulation (effectively larger batch with less memory)
gradient_accumulation_steps=4
```

### Model Not Learning (Loss Not Decreasing)

```python
# Try a higher learning rate
learning_rate=5e-5  # instead of 2e-5

# Ensure data is shuffled
train_data = dataset.shuffle(seed=42)

# Check for data issues
print(train_data[0])  # Inspect samples manually
```

### Accuracy Stuck at ~50% (Binary) or ~25% (4-class)

This usually means the model is random guessing. Check:

1. **Labels are correct** - Print a few examples to verify
2. **Learning rate too high** - Model weights are diverging
3. **Data preprocessing issue** - Tokenization may be wrong

---

## 📚 Additional Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [Fine-Tuning Best Practices](https://huggingface.co/docs/transformers/training)
- [Dataset Hub](https://huggingface.co/datasets) - Find more datasets to practice with

---

> 💡 **Pro Tip**: The best way to learn is by experimenting! Try changing hyperparameters, using different models (like `bert-base-uncased` or `roberta-base`), or training on the full datasets for better results.
