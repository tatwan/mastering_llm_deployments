# Module 02: Fine-Tuning & Transfer Learning

Welcome to the **Fine-Tuning** module! In this section, we move beyond using pre-trained models "out of the box" and learn how to adapt them to specific domains and tasks.

## 📚 Notebooks Overview

This module consists of 4 progressive notebooks designed to take you from the basics of Transfer Learning to advanced LLM Fine-Tuning using PEFT and LoRA.

### [01_transfer_learning.ipynb](./01_transfer_learning.ipynb)
**Topic:** Introduction to Transfer Learning with BERT
*   **Goal:** Understand the core concept of Transfer Learning—taking a model trained on a massive general dataset and adapting it to a specific task.
*   **Key Concepts:**
    *   **Pre-training vs. Fine-tuning:** The "Tennis vs. Squash" analogy.
    *   **Trainer API:** Hugging Face's high-level API to abstract the training loop.
    *   **Model Architecture:** How a classification head is added to a base Transformer.
*   **Use Case:** Binary Sentiment Classification (Positive/Negative).

### [02_sentiment_analysis.ipynb](./02_sentiment_analysis.ipynb)
**Topic:** Deep Dive into Sentiment Analysis (IMDB)
*   **Goal:** Build a production-ready sentiment analysis workflow, focusing on data preparation and evaluation.
*   **Key Concepts:**
    *   **Data Handling:** Dealing with variable text lengths and tokenization limits (512 tokens).
    *   **Evaluation Metrics:** Why "Accuracy" isn't enough. Introduction to **Precision**, **Recall**, and **F1-Score**.
    *   **Error Analysis:** Using Confusion Matrices to visualize where the model fails.
*   **Use Case:** Long-form Movie Review Classification.

### [03_summarization.ipynb](./03_summarization.ipynb)
**Topic:** Text Summarization with Seq2Seq Models
*   **Goal:** Switch from classification (Encoder-only) to generation (Encoder-Decoder) tasks.
*   **Key Concepts:**
    *   **Encoder-Decoder Architecture:** Using **T5** (Text-to-Text Transfer Transformer).
    *   **ROUGE Metrics:** How to evaluate generated text against reference summaries (N-gram overlap).
    *   **Generation Parameters:** Controlling output creativity with `temperature`, `beams`, and `sampling`.
*   **Use Case:** Dialogue Summarization (DialogSum dataset).

### [04_Fine_Tuning_LLM_Healthcare.ipynb](./04_Fine_Tuning_LLM_Healthcare.ipynb)
**Topic:** Fine-Tuning LLMs for Healthcare (PEFT & LoRA)
*   **Goal:** Fine-tune a modern Large Language Model (TinyLlama) on a specialized medical dataset using efficient techniques.
*   **Key Concepts:**
    *   **SFT (Supervised Fine-Tuning):** Teaching a chat model how to respond to instructions.
    *   **PEFT (Parameter-Efficient Fine-Tuning):** Updating only a tiny fraction (<1%) of parameters to save memory.
    *   **LoRA (Low-Rank Adaptation):** The specific adapter technique used for efficiency.
    *   **Quantization:** Loading models in 4-bit or 16-bit to fit on consumer hardware.
*   **Use Case:** Medical Question Answering (MedQuad dataset).

---

## 🔬 Technique Spotlight: Full Fine-Tuning vs. PEFT

You might notice a key difference in how we train the models across these notebooks:

### Notebooks 1-3: Full Fine-Tuning
*   **What we do:** We update **all** the parameters (weights) of the model.
*   **Models used:** BERT, DistilBERT, T5.
*   **Why possible:** These models are relatively "small" (60M - 220M parameters). They fit roughly into standard GPU memory even when we train every single neuron.
*   **Analogy:** Retraining an athlete for a new sport by working on every muscle group.

### Notebook 4: PEFT (Parameter-Efficient Fine-Tuning)
*   **What we do:** We freeze the massive main model and only train tiny "adapter" layers (less than 1% of parameters).
*   **Models used:** TinyLlama-1.1B (and larger LLMs like Llama-3-8B).
*   **Why necessary:** LLMs have billions of parameters. Full fine-tuning would require massive industrial clusters (hundreds of GPUs). PEFT allows us to do it on a single laptop.
*   **Analogy:** Putting a new specialized lens on a camera. You don't rebuild the camera; you just add a small attachment to change how it sees.

### ⚠️ Important Distinction: Fine-Tuning is NOT Training from Scratch

It is a common misconception that "Full Fine-Tuning" means starting from zero. It does not.

*   **Training from Scratch:** initializing weights randomly and training on billions of tokens to teach the model English, grammar, and reasoning. (Takes months/millions of dollars).
*   **Full Fine-Tuning:** starting with those **pre-trained weights** and simply adjusting them slightly. The model retains its foundational knowledge.

**Why is this better?**
1.  **Efficiency:** Fine-tuning takes hours/days, not months.
2.  **Data:** You need thousands of examples, not billions.
3.  **Knowledge:** The model already knows "English"; you are just teaching it "Medical English".

> [!WARNING]
> **Catastrophic Forgetting**: If you fine-tune *too* aggressively (e.g., high learning rate), the model might forget its original language skills. This is why we use low learning rates (e.g., `2e-5`) to gently nudge the weights.

## 🧠 Educational Deep Dive

### 1. Fine-Tuning vs. RAG (Retrieval-Augmented Generation)

A common question is: *"Should I fine-tune my model or just use RAG?"*

| Feature | **RAG (Retrieval-Augmented Gen)** | **Fine-Tuning** |
| :--- | :--- | :--- |
| **Analogy** | Taking an exam with an open textbook. | Going to medical school for 4 years. |
| **Primary Goal** | **Knowledge Retrieval**. Giving the model access to up-to-date facts (e.g., "What is the company's revenue in 2024?"). | **Behavior Adaptation**. Teaching the model a specific style, format, or jargon (e.g., "Speak like a pirate" or "Answer in JSON format"). |
| **Pros** | Cheap, easy to update facts, perfect for dynamic data. | specialized performance, lower latency (no retrieval step), better instruction following. |
| **Cons** | Limited by context window size. | Expensive to train, hard to unlearn facts, knowledge becomes outdated. |

**Verdict:** Use **RAG** for facts. Use **Fine-Tuning** for style, behavior, and specialized tasks. Often, the best systems use **both**.

### 2. Full Fine-Tuning vs. PEFT (LoRA)

| Feature | Full Fine-Tuning | PEFT (LoRA) |
| :--- | :--- | :--- |
| **What updates?** | All model parameters (e.g., 7 Billion). | A tiny adapter (e.g., 10 Million params). |
| **Memory Usage** | Massive. Requires 4-8 GPUs for large models. | Tiny. Can run on a single consumer GPU (or even CPU/Mac). |
| **Storage** | Checkpoints are huge (GBs). | Checkpoints are tiny (MBs). |
| **Performance** | The "Gold Standard" (slightly better). | Extremely close to full fine-tuning (usually >95% performance). |
| **Use Case** | When you have a cluster of H100s. | When you are a normal developer. |

### 3. Evaluation Metrics Cheatsheet

*   **Accuracy:** % of correct predictions. *Use for:* Balanced classification datasets.
*   **Precision:** "Quality". When it guesses Positive, is it actually Positive? *Use for:* Spam filters (don't delete real email).
*   **Recall:** "Quantity". Did it find all the Positives? *Use for:* Cancer detection (don't miss a case).
*   **F1-Score:** Harmonic balance of Precision and Recall. *Use for:* Most imbalanced real-world problems.
*   **ROUGE:** N-gram overlap. *Use for:* Summarization.
*   **BLEU:** Precision-based overlap. *Use for:* Translation.
