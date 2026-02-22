# GenreBlender  
### Generative AI Music Mixer & Genre Classifier

GenreBlender is an interactive AI system that generates cross-genre music and **quantifies how well the blend worked**.

Unlike typical generative demos, GenreBlender combines:

- 🎵 A generative music model  
- 🧠 A trained neural genre classifier  
- 📊 A measurable evaluation framework  

---

## 🌐 Live Demo

Deployed on Hugging Face Spaces:

👉 http://localhost:8501/  

---

## 🚀 Overview

GenreBlender allows users to:

- Select **two genres**
- Adjust a blending slider `α ∈ [0,1]`
- Generate a brand-new AI-composed track
- View the predicted genre probability distribution

We define the intended blend as:

```math
Target = (α · A) + ((1 − α) · B)
```

Our classifier predicts:

```math
P(genre_i | audio)
```

We compare the predicted distribution to the target — making genre blending:

- Interactive  
- Measurable  
- Controllable  
- Interpretable  

---

## 🛠 System Architecture

GenreBlender consists of two main components:

---

### 🎵 1. Generative Engine

- Built using **Meta MusicGen**
- Generates music from structured weighted prompts
- Example prompt:

> "A music track that is 70% classical and 30% hip hop."

---

### 🧠 2. Neural Genre Classifier

We trained a custom Multilayer Perceptron-based classifier to evaluate generated tracks.

#### 📊 Dataset
- GTZAN Genre Dataset

#### 🎧 Feature Extraction
- MFCCs
- RMS energy
- Harmonic/percussive features
- Tempo

#### 🔄 Preprocessing
- `StandardScaler`
- `LabelEncoder`
- `GroupShuffleSplit` (prevents data leakage)

#### 🏗 Model Architecture

```python
nn.Linear(input_size, 256)
ReLU
Dropout(0.3)

nn.Linear(256, 128)
ReLU
Dropout(0.3)

nn.Linear(128, 64)
ReLU
Dropout(0.3)

nn.Linear(64, num_classes)
```

- Trained for 100 epochs  
- Achieved **92% validation accuracy**

Classifier output uses:

```math
softmax(z_i) = e^{z_i} / Σ_j e^{z_j}
```

---

## 🔄 Full Pipeline

1. User selects Genre A and Genre B  
2. User sets blend weight `α`  
3. MusicGen generates audio  
4. Audio features are extracted  
5. Classifier predicts genre probabilities  
6. Results displayed in frontend  

This creates a **closed-loop controllable generation system**.

---

## 💻 Tech Stack

- Python  
- PyTorch  
- Scikit-learn  
- Librosa  
- Meta MusicGen  
- Streamlit / Flask  
- NumPy  
- Pandas  
- Joblib  
- Pydub  

---

## 📦 Installation (Local Setup)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/genreblender.git
cd genreblender
```

### 2️⃣ Create a virtual environment

```bash
conda create -n genreblender python=3.10
conda activate genreblender
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app

```bash
streamlit run app.py
```

The app will run on:

```
http://localhost:8501
```

---

## 📂 Required Model Files

The classifier relies on:

- `model_3sec.pth`
- `scaler.pkl`
- `label_encoder.pkl`

These ensure consistent preprocessing during inference.

---

## ⚠ Challenges

- CUDA compatibility with newer GPUs  
- Preventing data leakage in audio splits  
- Maintaining consistent preprocessing between training and inference  
- Optimizing generation speed for demo environments  

---

## 🎯 Key Contributions

- Built both generation and evaluation systems  
- Achieved 92% validation accuracy  
- Designed a measurable blending framework  
- Created an interpretable AI music system  

---

## 🔮 Future Improvements

- Embedding-level interpolation instead of prompt-only blending  
- KL divergence between target and predicted distributions  
- Real-time probability visualizations  
- More genres + larger datasets  
- Fully scalable GPU deployment  

---

## 📜 License

MIT License

---

## 👩‍💻 Authors
Prisha Priyadarshini, Srijan Roy Choudhury, Aaheli Rathi, Anjana Rao

Developed as part of a Generative AI project exploring controllable music synthesis and evaluation.

If you use this project or build on it, please consider giving it a ⭐ on GitHub.
