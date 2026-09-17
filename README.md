# Deep Learning-Based Fake News Detection Using CNN and LSTM
**TRUTHSCAN AI**

An end-to-end deep learning system for classifying news articles as Real or Fake using Convolutional Neural Networks (TextCNN) and Long Short-Term Memory networks (LSTM) built with PyTorch and standard full-stack web technologies.

---

## 1. Project Title
**Deep Learning-Based Fake News Detection Using CNN and LSTM**  
*Application Name:* TRUTHSCAN AI

---

## 2. Project Overview
TRUTHSCAN AI is an academic natural language processing (NLP) application engineered to classify the credibility and veracity of news articles. By leveraging two foundational deep learning paradigms—convolutional feature extraction and recurrent sequence modeling—the system offers comparative analysis, empirical evaluation metrics, and real-time interactive predictions through a modern, responsive web application.

The project runs on a single unified codebase supporting both local execution (`python server.py`) and cloud deployment (Vercel Serverless Functions) with zero dependency on Streamlit, Flask, or FastAPI.

---

## 3. Problem Statement
The exponential growth of digital media and uncurated news distribution channels has accelerated the spread of disinformation. False and misleading articles distort public discourse, polarize communities, and compromise decision-making. Automated detection systems that accurately identify subtle linguistic, semantic, and structural indicators of fake news are critical to maintaining digital media integrity.

---

## 4. Objectives
- Implement a robust text preprocessing and tokenization pipeline with zero data leakage (leakage-free sanitization removing publisher tags and dateline patterns).
- Build and train a multi-kernel **TextCNN** classifier to detect salient n-gram phrases and local rhetorical patterns.
- Build and train an **LSTM** recurrent neural network to capture long-range contextual dependencies and narrative flow.
- Ensure strict evaluation on an untouched test split using standard metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
- Provide a modern, clean web dashboard built with semantic HTML5, Vanilla CSS, and JavaScript for interactive article verification, side-by-side model comparison, and training management.

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
Place the dataset CSV or Excel files into `data/` (or `archive/`):
```
data/
├── Fake.csv (or Fake.xlsx)
└── True.csv (or True.xlsx)
```
Both CSV and Excel formats are natively recognized and loaded.

### Manual Download:
Download `Fake.csv` and `True.csv` from Kaggle:
- [ISOT Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### Automated Download:
You can automatically retrieve the authentic files using the built-in downloader:
```bash
python main.py --download-data
```
Or trigger download in the web interface under the **Training** tab.

---

## 7. Project Structure
```
fake_news_detection/
│
├── api/
│   └── index.py                      # Unified REST API handler (Localhost & Vercel Serverless)
│
├── public/                           # Frontend static assets
│   ├── index.html                    # Semantic HTML5 single-page application (5 tabs)
│   ├── css/
│   │   └── styles.css                # Dark AI metallic theme & responsive design system
│   └── js/
│       └── app.js                    # Client-side routing, API communication, and state management
│
├── data/
│   ├── Fake.csv                      # ISOT Fake articles (label 1)
│   ├── True.csv                      # ISOT Real articles (label 0)
│   ├── dataset_stats.json            # Cached dataset statistics
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
│   ├── preprocessing.py              # Text cleaning, sanitization, and combining
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
├── server.py                         # Localhost Web Application Server (ThreadingHTTPServer)
├── vercel.json                       # Vercel deployment configuration
├── config.py                         # Hyperparameters, directories, seeds
├── main.py                           # Command-line interface entry point
├── requirements.txt                  # Python dependencies (Torch, Scikit-Learn, Pandas, etc.)
├── run.bat                           # Double-click Windows launcher
└── README.md                         # Comprehensive documentation
```

---

## 8. Technologies
- **Python 3.9+**
- **PyTorch (torch >= 2.0.0):** Deep learning network modeling, backpropagation, CUDA/CPU tensor computations.
- **Scikit-Learn (>= 1.3.0):** Stratified data splitting, evaluation metrics (accuracy, precision, recall, F1, confusion matrices).
- **Matplotlib (>= 3.7.0):** Generation of loss curves, accuracy plots, and confusion matrix visualizations.
- **Pandas & NumPy:** Tabular data processing and numerical arrays.
- **Tqdm:** Terminal progress monitoring.
- **Frontend:** Semantic HTML5, Vanilla CSS3 (Custom Dark AI Metallic Design System), Vanilla JavaScript (ES6+).
- **Server:** Python Standard Library `http.server.ThreadingHTTPServer` (Localhost) & Vercel Serverless Functions (`api/index.py`).

---

## 9. CNN Architecture Explanation
**Convolutional Neural Network for Text (TextCNN):**
- *Principle:* Detects important local patterns, phrases, and n-grams in text.
- *Mechanism:* TextCNN applies parallel 1D convolutional filter banks of varying kernel sizes ($k = 3, 4, 5$) across the sequence of word embeddings. Each filter detects distinct n-gram combinations (tri-grams, four-grams, five-grams) commonly associated with sensationalist phrasing, emotional appeals, or formal journalistic reporting.
- *Pooling:* 1D Global Max-Pooling extracts the most prominent feature from each filter channel, creating translation-invariant representations.
- *Classification Head:* Pooled features are concatenated, passed through a two-stage MLP projection head (`Linear(num_filters * len(kernels), 128)` -> `ReLU` -> `Dropout(0.5)` -> `Linear(128, 1)`) with a sigmoid activation for binary classification.

---

## 10. LSTM Architecture Explanation
**Long Short-Term Memory (LSTM):**
- *Principle:* Processes the sequence of words and learns relationships between earlier and later parts of the text.
- *Mechanism:* Recurrent neural networks suffer from vanishing gradients over long sequences. LSTMs introduce an internal memory cell governed by three gating mechanisms:
  1. **Forget Gate:** Decides which prior historical information to discard.
  2. **Input Gate:** Determines which new incoming word semantics to store.
  3. **Output Gate:** Controls what information from the memory cell is emitted into the hidden state.
- *Classification Head:* The final hidden state vector ($h_n$) synthesizes the contextual meaning of the entire article, which is passed through a two-stage MLP head with dropout ($p = 0.3$) and classified via a dense linear layer.

---

## 11. Installation
Clone or navigate to the project directory, then execute the following Windows commands:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 12. Running the Web Application Locally
Start the local server using `server.py`:
```cmd
python server.py
```
Or use the CLI flag:
```cmd
python main.py --serve
```
Open your web browser and navigate to:
```
http://localhost:5000
```

---

## 13. Running Using `run.bat`
On Windows, you can start the entire project by simply double-clicking `run.bat`.

The batch file automatically:
1. Locates the project directory dynamically.
2. Checks for or creates the virtual environment (`venv`).
3. Installs any missing dependencies from `requirements.txt`.
4. Starts the TRUTHSCAN AI web application on `http://localhost:5000`.
5. Keeps the terminal open if an error occurs so you can inspect error logs.

---

## 14. Deploying to Vercel
The project is preconfigured for deployment on Vercel:
1. Ensure your repository contains `vercel.json`, `api/index.py`, and `public/`.
2. Push your code to GitHub.
3. Import the repository into your [Vercel Dashboard](https://vercel.com).
4. Vercel automatically detects `vercel.json` and routes:
   - Static assets (`/css/*`, `/js/*`, `index.html`) via `public/`.
   - API calls (`/api/*`) to `api/index.py`.
5. *(Optional for Serverless PyTorch limits):* If PyTorch bundle size exceeds Vercel Serverless quotas (250MB uncompressed limit), set the environment variable `ML_BACKEND_URL` in Vercel settings pointing to a dedicated backend instance (e.g. Railway, Render, or self-hosted GPU). The API handler seamlessly proxies prediction calls.

---

## 15. Training
You can train models via the web app under the **Training Session** tab, or using the CLI:

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

## 16. Evaluation
Evaluate the saved checkpoints on the untouched 10% test dataset split:
```cmd
python main.py --evaluate
```
This computes genuine metrics, saves `results/model_comparison.csv`, and generates visual charts in `results/plots/`.

### Benchmark Results on ISOT Test Split:
| Model | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **TextCNN** | **95.00%** | **0.9359** | **0.9563** | **0.9460** |
| **LSTM** | **91.20%** | **0.9843** | **0.8210** | **0.8952** |

---

## 17. Hardware Requirements
- **Operating System:** Windows 10/11, Linux, or macOS
- **Processor:** Intel Core i5 / AMD Ryzen 5 or higher
- **RAM:** Minimum 8 GB (16 GB recommended)
- **Disk Space:** 500 MB free disk space (including dataset)
- **GPU (Optional):** NVIDIA GPU with CUDA support. If not present, the system automatically runs on CPU.

---

## 18. Troubleshooting
- **Dataset Not Found:** Ensure `data/Fake.csv` and `data/True.csv` exist or run `python main.py --download-data`.
- **Model Not Trained Yet:** Navigate to the **Training Session** page in the web app or run `python main.py --train both --mode quick`.
- **Out of Memory (OOM):** Use `--mode quick` which reduces batch size to 16 and sequence length to 200 tokens.
- **Port Conflict (5000):** Launch with a custom port: `set PORT=5001 && python server.py`.

---

## 19. Limitations & Future Scope
- **Limitations:** Predictions are based on stylistic, lexical, and structural patterns learned from the ISOT dataset. The models do not query live search engines or verify breaking events.
- **Future Work:** Multi-source cross-dataset evaluation (e.g. LIAR, FA-KES), Bidirectional LSTM (BiLSTM), and Attention/Explainability heatmaps.
