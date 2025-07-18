## ML-Models: Spam Detection & Face Alignment
This repository showcases two machine learning systems built from scratch as part of a university project. Both were designed with real-world relevance in mind and follow industry best practices like modular design, feature engineering, and model benchmarking.

---

### Overview

#### Task 1 – Spam Detection (Classification)
Developed a classical machine learning pipeline to detect whether a message is spam or not. This included:

- Data analysis & visualisation (e.g. word clouds, class imbalance)
- Text preprocessing: cleaning, tokenisation, stop-word removal
- Feature extraction: exclamation mark density, vocab richness, marketing keywords, etc.
- Model comparison: Logistic Regression, Decision Tree, Naive Bayes, SVC, Random Forest
- Final model: **Random Forest**, selected based on precision, recall, accuracy & F1 score

*Machine Learning Pipeline:*
![alt text](https://github.com/finsherwell/ML-Models/blob/main/docs/flowcharts/spam_detection_design.PNG "Spam Detection Design Pipeline")

#### Task 2 – Face Alignment (Regression)
Designed a pipeline to locate facial landmarks (e.g., eyes, nose, mouth) on face images:

- Image preprocessing: grayscale conversion, resizing, normalisation
- Feature engineering: SIFT descriptors, edge maps (Canny), pixel intensity stats
- Model comparison: Linear Regression, Ridge Regression, SVR, Random Forest
- Final model: **Linear Regression**, chosen for lowest MSE and highest R² score

*Machine Learning Pipeline:*
![alt text](https://github.com/finsherwell/ML-Models/blob/main/docs/flowcharts/face_alignment_design.PNG "Face Alignment Design Pipeline")

---

### Directory Structure
```
ML-Models/
├── data/
│   └── task1/, task2/         # Raw and processed datasets (not included in repo)
├── notebooks/
│   └── task1/, task2/         # Jupyter notebooks for each stage
├── reports/
│   └── Report.pdf             # Project report with design, results & analysis
├── src/
│   └── ...                    # Reusable Python scripts (feature extraction, model eval, etc.)
├── requirements.txt
├── config.yaml
└── README.md
```

---

### Build Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/finsherwell/ML-Models.git
   cd ML-Models
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv/Scripts/activate on Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Data Files**:
   - For Spam Detection: place CSVs in `data/task1/raw/`
   - For Face Alignment: place `.npz` files in `data/task2/raw/`

5. **Run Notebooks**:
   - Launch JupyterLab or VSCode with the correct Python kernel selected (Python 3.13.3 used).
   - Run notebooks in order using `Run All`.

---

### Results

#### Spam Detection:
- **Best Model**: Random Forest
- **Key Metrics**: Highest precision, recall, and F1 score across models

#### Face Alignment:
- **Best Model**: Linear Regression
- **Key Metrics**: Lowest MSE, strong visual alignment in predictions

---

### Lessons Learned
- Simpler models like Logistic or Linear Regression can outperform complex ones when paired with well-crafted features.
- Feature engineering is vital: marketing keywords and punctuation patterns for spam; edge detection and SIFT for face alignment.
- Data quality and consistency in preprocessing have a huge impact on outcomes.

#### Disclaimer:
I have tried to include the diagrams from the data analysis booklets where possible so you can see for yourself. However, some have been hidden for intellectual property and privacy reasons. If you wish to see more as to how the notebooks work, I encourage you to clone the repository and try for yourself. I am not liable for any copyright issues regarding datasets - this was a university assignment. If you have any issues please contact me.
