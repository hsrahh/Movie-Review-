# 🎬 Text Sentiment Analysis using Deep Learning (LSTM)

A Deep Learning project that classifies movie reviews as **POSITIVE** or **NEGATIVE** using an LSTM neural network — trained on the IMDB dataset with ~87% accuracy.

---

## 📌 Project Overview

This project builds an end-to-end NLP pipeline using:
- **Word Embeddings** to convert words into meaningful vectors
- **LSTM (Long Short-Term Memory)** to understand sequential context
- **Dropout** to prevent overfitting
- **Binary Classification** to output a sentiment probability

> Built as a learning project to deeply understand every component of a Deep Learning pipeline — not just run the code, but know *why* each line exists.

---

## 🧠 Model Architecture

```
Input (raw text)
     ↓
Embedding Layer   →  vocab=10,000 | dim=128  | output: (batch, 200, 128)
     ↓
LSTM Layer        →  64 units     | dropout=0.2              output: (batch, 64)
     ↓
Dropout Layer     →  rate=0.5     (prevents overfitting)
     ↓
Dense Layer       →  1 neuron     | activation=sigmoid       output: (batch, 1)
     ↓
Prediction:  > 0.5 = POSITIVE ✅  |  < 0.5 = NEGATIVE ❌
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Dataset | IMDB Movie Reviews (built into Keras) |
| Total samples | 50,000 reviews |
| Training set | 25,000 reviews |
| Test set | 25,000 reviews |
| Classes | Positive (1) / Negative (0) |
| Class balance | 50% / 50% (perfectly balanced) |

---

## ⚙️ Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| `VOCAB_SIZE` | 10,000 | Top 10k most frequent words |
| `MAX_LENGTH` | 200 | Max words per review |
| `EMBEDDING_DIM` | 128 | Size of each word vector |
| `LSTM_UNITS` | 64 | Memory cells in LSTM |
| `DROPOUT_RATE` | 0.5 | 50% neurons off during training |
| `BATCH_SIZE` | 64 | Reviews per weight update |
| `EPOCHS` | 10 (early stopping) | Max training passes |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-lstm.git
cd sentiment-analysis-lstm
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python sentiment_analysis_deep_learning.py
```

The IMDB dataset (~17MB) will be downloaded automatically on first run.

---

## 📈 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~87% |
| Test Loss | ~0.31 |

Training stops automatically via **Early Stopping** when validation loss stops improving (patience = 3 epochs).

---

## 💡 Key Concepts Covered

- ✅ **Tokenization & Padding** — converting text to fixed-length integer sequences
- ✅ **Word Embeddings** — learning semantic word representations
- ✅ **LSTM & Gates** — forget gate, input gate, cell state, output gate
- ✅ **Dropout Regularization** — preventing overfitting
- ✅ **Binary Cross-Entropy Loss** — loss function for 2-class problems
- ✅ **Adam Optimizer** — adaptive gradient descent
- ✅ **Early Stopping** — halting training at the right time
- ✅ **Training Curves** — reading accuracy/loss plots

---

## 📁 Project Structure

```
sentiment-analysis-lstm/
│
├── sentiment_analysis_deep_learning.py   # Main project file (fully commented)
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

---

## 🔮 Future Improvements

- [ ] Bidirectional LSTM (read sequence forwards + backwards)
- [ ] Pre-trained GloVe / Word2Vec embeddings
- [ ] Multi-layer LSTM (deeper network)
- [ ] Fine-tune BERT for higher accuracy
- [ ] Deploy as a web app using Flask / Streamlit

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?style=flat&logo=keras)
![NumPy](https://img.shields.io/badge/NumPy-1.23+-lightblue?style=flat&logo=numpy)

---

## 👤 Author

**Harsh Joil**  
📧 harshjoil02@gmail.com 
🔗 [LinkedIn](https://www.linkedin.com/in/harsh-joil-0bbb3227a/)  
🐙 [GitHub](https://github.com/hsrahh)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ **If you found this helpful, please give it a star!** It motivates me to keep building and sharing.
