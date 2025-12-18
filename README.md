# Tweet Sentiment Phrase Extraction

## Project Overview

This project implements a fine-tuned RoBERTa model for tweet sentiment phrase extraction, addressing the interpretability challenge in sentiment analysis by identifying specific text spans that justify given sentiment labels.

**Project Details:**
- **Course**: 62FIT4ATI - Fall 2025
- **Group**: 3
- **Topic**: 4 - Tweet Sentiment Phrase Extraction
- **Model Architecture**: RoBERTa-base with custom span prediction head
- **Performance**: Jaccard Score ~0.705 (approaching human benchmark of 0.78)

## Problem Description

Unlike traditional sentiment classification, this project tackles **span extraction** - identifying the specific phrase in a tweet that justifies its sentiment label. This provides interpretability by explaining *why* a text has a particular sentiment.

**Examples:**
- Tweet: "The food was amazing but the service was terrible."
- Positive sentiment → Selected text: "amazing"
- Negative sentiment → Selected text: "terrible"

## Dataset

- **Size**: ~27,500 training samples, ~3,500 test samples
- **Format**: CSV files with tweet text, sentiment labels, and target phrases
- **Source**: [Google Drive Dataset](https://drive.google.com/drive/folders/1b4KvfO_Vid9HputdJwATAQBmYImWhmgR?usp=share_link)

## Requirements

### Dependencies
```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn tqdm
```

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hihiririri/PROJECT-4-TWEET-SENTIMENT-PHRASE-EXTRACTION
   cd PROJECT-4-TWEET-SENTIMENT-PHRASE-EXTRACTION
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Access the dataset from the Google Drive link above
   - Place CSV files in the project directory

4. **Run the notebook**:
   ```bash
   jupyter notebook 62FIT4ATI_Group_3_Topic_4.ipynb
   ```

## Model Architecture

- **Base Model**: RoBERTa-base (125M parameters)
- **Custom Head**: Span prediction layer for start/end position prediction
- **Input Format**: `<s> sentiment </s> </s> tweet_text </s>`
- **Output**: Start and end token positions for phrase extraction

## Key Features

### Optimization Techniques
1. **AdamW Optimizer** with learning rate 3e-5 and weight decay 0.01
2. **Linear Warmup Scheduler** with 100 warmup steps
3. **Mixed Precision Training** (FP16) using GradScaler  
4. **Gradient Clipping** (implicit through mixed precision)
5. **Early Stopping** after 3 epochs with model checkpointing

### Advanced Capabilities
- **Adaptive Strategy**: Different behavior for neutral vs. positive/negative sentiments
- **Tokenizer**: RobertaTokenizerFast for improved performance
- **Max Sequence Length**: 96 tokens
- **Batch Size**: 32
- **Character-level Accuracy**: Uses offset mapping for precise span extraction

## Performance Results

## Performance Results

### Jaccard Scores by Sentiment
| Sentiment Category | Jaccard Score | Analysis |
|-------------------|---------------|----------|
| **Neutral** | **0.970** | Near-perfect "copy-all" strategy |
| **Positive** | **0.597** | Selective phrase extraction |
| **Negative** | **0.512** | Most challenging due to annotation subjectivity |

### Training Progress
- **Epochs**: 3
- **Final Training Loss**: 1.3486 (Epoch 3)
- **Model Checkpoints**: Saved after each epoch
- **Training Time**: ~2.5 minutes per epoch on GPU

### Key Insights
- **Neutral Strategy**: Near-perfect "copy-all" behavior (0.97 Jaccard)
- **Emotional Strategy**: Selective extraction focusing on sentiment-bearing phrases
- **Human Benchmark**: 0.78 Jaccard score on this dataset

## File Structure

```
PROJECT-4-TWEET-SENTIMENT-PHRASE-EXTRACTION/
├── 62FIT4ATI_Group_3_Topic_4.ipynb    # Main notebook with implementation
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/                              # Dataset folder
│   ├── train.csv                      # Training data (~27,500 samples)
│   └── test.csv                       # Test data (~3,500 samples)
├── models/                            # Trained model checkpoints
│   ├── roberta_epoch_1.bin            # Model checkpoint (epoch 1)
│   ├── roberta_epoch_2.bin            # Model checkpoint (epoch 2)
│   └── roberta_epoch_3.bin            # Model checkpoint (epoch 3)
├──  Tweet_Sentiment_Analysis_Report.docx
│    
└── .gitattributes                     # Git LFS configuration for model files


## Usage

### Training
The model is trained using the notebook with the following key steps:
1. Data preprocessing and tokenization
2. Custom dataset creation with offset mapping
3. Model training with optimization techniques
4. Validation and performance evaluation

### Inference
```python
# Load the trained model
model.load_state_dict(torch.load("roberta_epoch_3.bin"))

# Predict on new data
prediction = predict_tweet("The food was absolutely amazing!", "positive", model, tokenizer, device)
print(prediction)  # Output: "absolutely amazing"
```

## Key Challenges Addressed

1. **Subjectivity**: Human annotation inconsistencies
2. **Noisy Data**: Twitter text with slang, abbreviations, emojis
3. **Behavioral Differences**: Neutral vs. emotional sentiment extraction strategies
4. **Class Imbalance**: Different extraction patterns across sentiment types

## Future Improvements

1. **Ensemble Learning**: Combine multiple transformer models
2. **Advanced Post-processing**: Intelligent boundary artifact removal
3. **Data Augmentation**: Back-translation for positive/negative classes
4. **Error Analysis**: Detailed failure case investigation

## Authors

**Group 3 Members:**
- Do Dinh Thuc, Nguyen Thi Kim Hue, Dong Duy Dong 
- Course: 62FIT4ATI, Fall 2025
- Hanoi University, Faculty of Information Technology

## License

This project is developed for educational purposes as part of the 62FIT4ATI course curriculum.

## Acknowledgments

- RoBERTa model by Facebook AI Research
- Hugging Face Transformers library
- Course instructors and teaching assistants
