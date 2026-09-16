# Deep Learning-Based Fake News Detection Using CNN and LSTM
**TRUTHSCAN AI**

An end-to-end deep learning system for classifying news articles as Real or Fake using Convolutional Neural Networks (TextCNN) and Long Short-Term Memory networks (LSTM) built with PyTorch and Streamlit.

---

## 1. Project Title
**Deep Learning-Based Fake News Detection Using CNN and LSTM**  
*Application Name:* TRUTHSCAN AI

---

## 2. Project Overview
TRUTHSCAN AI is an academic natural language processing (NLP) application engineered to classify the credibility and veracity of news articles. By leveraging two foundational deep learning paradigms—convolutional feature extraction and recurrent sequence modeling—the system offers comparative analysis, empirical evaluation metrics, and real-time interactive predictions.

---

## 3. Problem Statement
The exponential growth of digital media and uncurated news distribution channels has accelerated the spread of disinformation. False and misleading articles distort public discourse, polarize communities, and compromise decision-making. Automated detection systems that accurately identify subtle linguistic, semantic, and structural indicators of fake news are critical to maintaining digital media integrity.

---

## 4. Objectives
- Implement a robust text preprocessing and tokenization pipeline with zero data leakage.
- Build and train a multi-kernel **TextCNN** classifier to detect salient n-gram phrases and local rhetorical patterns.
- Build and train an **LSTM** recurrent neural network to capture long-range contextual dependencies and narrative flow.
- Ensure strict evaluation on an untouched test split using standard metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
- Provide a modern, clean Streamlit web dashboard for interactive article verification, side-by-side model comparison, and training management.

---

## 5. Dataset
The project is built on the benchmark **ISOT Fake and Real News Dataset** from the University of Victoria:
- **True.csv:** 21,417 authentic news articles crawled from Reuters.com (Target label: `0` / Real).
- **Fake.csv:** 23,481 unverified and debunked articles from PolitiFact and Wikipedia (Target label: `1` / Fake).
- **Attributes:**
  - `title`: Article headline
  - `text`: Main body text
  - `subject`: Domain category (Political News, World News, Government, etc.)
  - `date`: Publication timestamp

---

## 6. Dataset Download Instructions
Place the dataset CSV or Excel files into `archive/` or `data/`:
```
archive/ (or data/)
├── Fake.csv (or Fake.xlsx)
└── True.csv (or True.xlsx)
```
Both CSV and Excel formats are natively recognized and loaded.

### Manual Download:
Download `Fake.csv` and `True.csv` from Kaggle or GitHub:
- [ISOT Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### Automated Download:
You can also automatically retrieve the authentic files using the built-in downloader:
```bash
python main.py --download-data
```
Or click **Download ISOT Dataset Automatically** in the web interface under the **Training** tab.

---

## 7. Project Structure
```
fake_news_detection/
│
├── data/
│   ├── Fake.csv                      # ISOT Fake articles (label 1)
│   ├── True.csv                      # ISOT Real articles (label 0)
│   └── processed/
│       └── splits.npz                # Stratified splits (train/val/test)
│
├── checkpoints/
│   ├── cnn_best.pth                  # Best TextCNN PyTorch model
│   └── lstm_best.pth                 # Best LSTM PyTorch model
│
├── tokenizer/
│   └── vocabulary.json               # Word-to-index mapping (train split only)
│
├── models/
│   ├── textcnn_model.py              # TextCNN PyTorch architecture
│   └── lstm_model.py                 # LSTM PyTorch architecture
│
├── src/
│   ├── preprocessing.py              # Text cleaning and combining
│   ├── tokenizer.py                  # Tokenizer, padding, and truncation
│   ├── dataset.py                    # Stratified loading & PyTorch Dataset
│   ├── download_data.py              # Authentic dataset downloader utility
│   ├── train.py                      # Training loop & validation tracking
│   ├── evaluate.py                   # Test evaluation & chart generation
│   └── predict.py                    # Inference and confidence calculation
│
├── results/
│   ├── model_comparison.csv          # Real test evaluation metrics
│   ├── cnn_history.json              # CNN loss/accuracy per epoch
│   ├── lstm_history.json             # LSTM loss/accuracy per epoch
│   └── plots/
│       ├── loss_accuracy_curves.png  # Training vs validation learning curves
│       ├── confusion_matrices.png    # Confusion matrix heatmaps
│       └── model_comparison.png      # Side-by-side metric comparison bar chart
│
├── app.py                            # Streamlit multi-page web application
├── config.py                         # Hyperparameters, directories, seeds
├── main.py                           # Command-line interface entry point
├── requirements.txt                  # Python dependencies
├── run.bat                           # Double-click Windows launcher
└── README.md                         # Comprehensive documentation
```

---

## 8. Technologies
- **Python 3.9+**
- **PyTorch (torch >= 2.0.0):** Deep learning network modeling, backpropagation, CUDA/CPU tensor computations.
- **Streamlit (>= 1.30.0):** Interactive academic dashboard.
- **Scikit-Learn (>= 1.3.0):** Stratified data splitting, evaluation metrics (accuracy, precision, recall, F1, confusion matrices).
- **Matplotlib (>= 3.7.0):** Generation of loss curves, accuracy plots, and confusion matrix visualizations.
- **Pandas & NumPy:** Tabular data processing and numerical arrays.
- **Tqdm:** Terminal progress monitoring.

---

## 9. CNN Architecture Explanation
**Convolutional Neural Network for Text (TextCNN):**
- *Principle:* Detects important local patterns, phrases, and n-grams in text.
- *Mechanism:* TextCNN applies parallel 1D convolutional filter banks of varying kernel sizes ($k = 3, 4, 5$) across the sequence of word embeddings. Each filter detects distinct n-gram combinations (tri-grams, four-grams, five-grams) commonly associated with sensationalist phrasing, emotional appeals, or formal journalistic reporting.
- *Pooling:* 1D Global Max-Pooling extracts the most prominent feature from each filter channel, creating translation-invariant representations.
- *Classification:* Pooled features are concatenated, passed through a dropout regularization layer ($p = 0.5$), and projected through a dense linear layer with a sigmoid activation for binary classification.

---

## 10. LSTM Architecture Explanation
**Long Short-Term Memory (LSTM):**
- *Principle:* Processes the sequence of words and learns relationships between earlier and later parts of the text.
- *Mechanism:* Recurrent neural networks suffer from vanishing gradients over long sequences. LSTMs introduce an internal memory cell governed by three gating mechanisms:
  1. **Forget Gate:** Decides which prior historical information to discard.
  2. **Input Gate:** Determines which new incoming word semantics to store.
  3. **Output Gate:** Controls what information from the memory cell is emitted into the hidden state.
- *Classification:* The final hidden state vector ($h_n$) synthesizes the contextual meaning of the entire article, which is then regularized with dropout ($p = 0.3$) and classified via a dense linear layer.

---

## 11. Installation
Clone or navigate to the project directory, then execute the following Windows commands:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 12. Training
You can train models via the Streamlit web app or using the CLI:

### Quick Mode (Recommended for Standard Laptops):
Uses 5,000 authentic ISOT articles, 2 epochs, batch size 16:
```cmd
python main.py --train both --mode quick
```

### Normal Mode:
Uses 20,000 authentic ISOT articles, 5 epochs, batch size 32:
```cmd
python main.py --train both --mode normal
```

### Training Individual Models:
```cmd
python main.py --train cnn --mode quick
python main.py --train lstm --mode quick
```

---

## 13. Evaluation
Evaluate the saved checkpoints on the untouched 10% test dataset split:
```cmd
python main.py --evaluate
```
This computes genuine metrics, saves `results/model_comparison.csv`, and generates visual charts in `results/plots/`.

---

## 14. Running Streamlit
Start the web dashboard locally:
```cmd
streamlit run app.py
```
Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 15. Running Using `run.bat`
On Windows, you can start the entire project by simply double-clicking `run.bat`.

The batch file automatically:
1. Locates the project directory dynamically.
2. Checks for or creates the virtual environment (`venv`).
3. Installs any missing dependencies from `requirements.txt`.
4. Starts the Streamlit application on `http://localhost:8501`.
5. Keeps the terminal open if an error occurs so you can inspect error logs.

---

## 16. Hardware Requirements
- **Operating System:** Windows 10/11, Linux, or macOS
- **Processor:** Intel Core i5 / AMD Ryzen 5 or higher
- **RAM:** Minimum 8 GB (16 GB recommended)
- **Disk Space:** 500 MB free disk space (including dataset)
- **GPU (Optional):** NVIDIA GPU with CUDA support. If not present, the system automatically runs on CPU.

---

## 17. Troubleshooting
- **Dataset Not Found:** Ensure `data/Fake.csv` and `data/True.csv` exist or run `python main.py --download-data`.
- **Model Not Trained Yet:** Navigate to the **Training** page in the web app or run `python main.py --train both --mode quick`.
- **Out of Memory (OOM):** Use `--mode quick` which reduces batch size to 16 and sequence length to 200 tokens.
- **Port Conflict (8501):** Launch with a custom port: `streamlit run app.py --server.port 8502`.

---

## 18. Limitations
- Predictions are based on stylistic, lexical, and structural patterns learned from the ISOT dataset (articles from 2016–2017).
- The models do not query live external search engines or perform real-time verification of breaking events.
- Domain shifts in modern journalistic styles may require retraining on newer corpora.

---

## 19. Future Improvements
- Multi-source training incorporating datasets such as LIAR or FA-KES.
- Bidirectional LSTM (BiLSTM) and Attention mechanisms for word importance visualization.
- Explainability integration using Integrated Gradients or LIME to highlight deceptive phrases.
