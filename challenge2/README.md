<h1 align="center">🌱 Soil Image Classification Challenge (Binary)</h1>

<p align="center">
  <img src="https://github.com/sanskaryo/Soil_classification_project_annam/blob/main/challenge2/docs/project_image.jpg?raw=true" alt="Project Image" width="500"/>
</p>

<p align="center">
  <strong>🔥 F1 Score: 0.8989 (Public) | Private Rank: 48</strong><br>
  🧠 Solo Participant | Finalist at <strong>Annam.ai × IIT Ropar</strong>
</p>

---

## 🧾 Overview

This repository contains my binary classifier solution for the Soil Image Classification Challenge organized by [Annam.ai](https://www.annam.ai/) and IIT Ropar. The goal is to predict whether an input image contains soil or not using CNN-based computer vision techniques.

**Why it matters:**
- 🌾 Precision agriculture & crop health
- ⛰️ Geological mapping & terrain analysis
- 🌍 Environmental monitoring & sustainability

---

## 📊 Leaderboard Performance

| Metric       | Public Score | Private Rank |
|--------------|--------------|--------------|
| 🔗 F1 Score  | 0.8989       | 48           |

---

## 🏁 Competition Details

| Detail          | Description                                 |
|-----------------|---------------------------------------------|
| **Organizer**   | Annam.ai × IIT Ropar                        |
| **Task**        | Binary classification (Soil / Non-Soil)    |
| **Deadline**    | May 25, 2025, 11:59 PM IST                  |
| **Evaluation**  | F1 Score (harmonic mean of Precision & Recall) |
| **Final Status**| Solo Submission, Finalist                   |

---

## 🧠 Model Pipeline

```mermaid
graph TD
  A[Raw Images] --> B[Preprocessing]
  B --> C[Train/Val Split + Augmentation]
  C --> D[Model (EfficientNet/Baseline)]
  D --> E[Inference]
  E --> F[Threshold Tuning]
  F --> G[Final Predictions CSV]
📁 Project Structure
text
Copy code
challenge2/
├── data/                # Dataset & synthetic 'Not Soil' images
├── docs/cards/          # Diagrams & cards
│   └── architecture.png # Model architecture
├── notebooks/           # Jupyter notebooks
│   ├── training.ipynb   # Model training workflow
│   └── inference.ipynb  # Inference & submission
├── src/                 # Processing scripts
│   ├── preprocessing.py # Data augmentation & synthetic images
│   └── postprocessing.py# Threshold tuning & metrics
├── download.sh          # Data download script
├── requirements.txt     # Python dependencies
└── README.md            # This file
🏋️‍♂️ Training Highlights
Input Size: 224×224 px

Augmentations: RandomFlip, Rotation, ColorJitter

Model Architectures: EfficientNet B0, ResNet variants

Loss Function: Binary Cross-Entropy

Optimization: Adam, learning rate scheduling

🧪 Evaluation & Thresholding
Metric: Macro F1-Score

Threshold Tuning: Grid search over [0.1, 0.9] to maximize validation F1.

python
Copy code
def tune_threshold(y_true, y_probs):
    thresholds = np.arange(0.1, 0.9, 0.01)
    # evaluate F1 at each thresh... return best
📌 Key Learnings
Data Augmentation significantly improved generalization.

EfficientNet performed robustly despite class imbalance.

Custom Thresholding was crucial to boost F1 score.

Modular Code ensures reproducibility and ease of experimentation.

🚀 Setup & Run
Clone repo

cmd
Copy code
git clone https://github.com/gogulgupta/Soil-classification-project.git
cd Soil-classification-project/challenge2
Install dependencies

cmd
Copy code
pip install -r requirements.txt
Download data

cmd
Copy code
bash download.sh
Prepare synthetic data

cmd
Copy code
python src/preprocessing.py
Train model

Open notebooks/training.ipynb, run all cells.

Run inference

Open notebooks/inference.ipynb, run all cells to generate submission.csv.


🚀 Author
Gogul Gupta – 3rd Year AIML Student
📍 Dr. A.P.J. Abdul Kalam Technical University | 👨‍💻 Building cool ML tools
🔗 Email: gogulguptaji@gmail.com

yaml
Copy code
