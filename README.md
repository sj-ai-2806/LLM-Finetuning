# README: Fine-Tuning with LoRA and QLoRA

This repository contains two Jupyter notebooks, `lora_peft.ipynb` and `qLora.ipynb`, which demonstrate the process of fine-tuning large language models (LLMs) using **LoRA** and **QLoRA** techniques for specific financial datasets. These notebooks utilize Hugging Face's Transformers and PEFT (Parameter-Efficient Fine-Tuning) libraries to efficiently fine-tune models with reduced memory and computational requirements.

## Contents

- **`lora_peft.ipynb`**: Implements fine-tuning of the Mistral-7B model using LoRA on a sentiment analysis dataset.
- **`qLora.ipynb`**: Implements fine-tuning of the Gemma-2B model using QLoRA on a QA dataset.

## Datasets

### 1. **Sentiment Analysis Dataset** (LoRA)
   - **Source**: [FinGPT Sentiment Analysis](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train?row=0)
   - **Purpose**: Used for training the Mistral-7B model to perform sentiment analysis in the financial domain.

### 2. **QA Dataset** (QLoRA)
   - **Source**: [FinGPT FIQA QA](https://huggingface.co/datasets/FinGPT/fingpt-fiqa_qa)
   - **Purpose**: Used for training the Gemma-2B model to perform question answering specifically tailored to financial contexts.

## Fine-Tuning Techniques

### LoRA (Low-Rank Adaptation)
LoRA is a parameter-efficient fine-tuning approach that introduces low-rank matrices to model fine-tuning. Instead of adjusting all weights in the model, LoRA injects small trainable matrices into the network that modify specific layers during fine-tuning, keeping the core model weights frozen. This approach significantly reduces computational requirements, making it ideal for scenarios where resources are limited.

   - **Notebook**: `lora_peft.ipynb`
   - **Model**: Mistral-7B
   - **Task**: Sentiment Analysis in the financial domain
   - **Dataset**: FinGPT Sentiment Analysis

### QLoRA (Quantized Low-Rank Adaptation)
QLoRA extends the LoRA method by quantizing the model to 4-bit precision, allowing even larger models to be fine-tuned within constrained memory. By reducing precision, QLoRA enables efficient fine-tuning on consumer-grade GPUs or limited-resource environments without sacrificing model performance. This approach is especially useful for large-scale question-answering models.

   - **Notebook**: `qLora.ipynb`
   - **Model**: Gemma-2B
   - **Task**: Question Answering in the financial domain
   - **Dataset**: FinGPT FIQA QA

## Requirements

To execute these notebooks, you will need:

- **Python 3.8+**
- **Hugging Face Transformers**
- **PEFT** (Parameter Efficient Fine-Tuning)
- **Datasets** from Hugging Face
- **CUDA** (for GPU acceleration, optional but recommended)

You can install the required libraries using:

```bash
pip install transformers peft datasets
```

## Usage

1. Clone the repository and open the notebooks in Jupyter Notebook or JupyterLab.
2. Run each notebook cell by cell. Ensure you have the necessary datasets available locally or are logged into Hugging Face to access them directly.
3. Customize model parameters as needed (learning rate, batch size, number of epochs, etc.) to optimize fine-tuning for your specific use case.

## Results

After fine-tuning, you will have two specialized models:

1. **LoRA Fine-Tuned Model**: A sentiment analysis model that can interpret financial text.
2. **QLoRA Fine-Tuned Model**: A question-answering model tailored to financial QA contexts.

These models can be further evaluated or deployed for applications requiring domain-specific sentiment analysis or question answering.

## References

- Hugging Face Datasets: [FinGPT Sentiment](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train?row=0) and [FinGPT FIQA QA](https://huggingface.co/datasets/FinGPT/fingpt-fiqa_qa)
- Hugging Face Transformers and PEFT Library
