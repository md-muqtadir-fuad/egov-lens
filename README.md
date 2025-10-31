# eGov-Lens: A Multi-Dimensional Machine Learning Approach to Aspect-Based Public Feedback Analysis on Bengali e-Government Platforms

This repository contains the official code and dataset for the paper **"eGov-Lens: A Multi-Dimensional Machine Learning Approach to Aspect-Based Public Feedback Analysis on Bengali e-Government Platforms"**.

eGov-Lens is a practical pipeline designed for the aspect-based analysis of public feedback on Bengali e-government channels. This project curates a new dataset and benchmarks classical machine learning models against transformer-based architectures to find the best-performing models in terms of both classification quality and computational cost.

## Features

* **Comprehensive Dataset:** A new, manually annotated dataset of **34,001 Bengali comments** collected from 2020-2024.
* **Multi-Aspect Analysis:** Comments are classified across four distinct, policy-relevant aspects: **Sentiment**, **Ease of Find (EoF)**, **Responsiveness**, and **Accuracy**.
* **Model Benchmarking:** We evaluate 7 different model combinations, comparing text representations like **CountVectorizer (BoW)**, **mBERT**, and **BanglaBERT** with classifiers like **Logistic Regression (LR)**, **Support Vector Classifier (SVC)**, and **Naive Bayes (NB)**.
* **Deployment-Aware Ranking:** We use an **Entropy-weighted TOPSIS** method to rank all models based on six criteria (Accuracy, F1, Recall, AUC, Training Time, Inference Time) to provide actionable guidance for deployment on resource-constrained infrastructure.

# Dataset

The primary dataset (`Merged Dataset` directory) contains 34,001 annotated user comments. The comments were collected from the official social media pages and public forums of 12 e-government services:

* Agriculture (DAE)
* BDRIS (Birth and Death Registration) 
* BRTA (Bangladesh Road Transport Authority)
* BTRC (Bangladesh Telecommunication Regulatory Commission)
* Consumer Rights
* Digital Bangladesh 
* Education 
* Election Commission 
* Medical Services
* Police Services
* Postal Service
* Teletalk

Each relevant comment is labeled for the four aspects mentioned above (Sentiment, EoF, Responsiveness, Accuracy).

# Models and Experiments

The project systematically evaluates seven primary model combinations using three text representation techniques and three classifiers. The baseline model for comparison is Count Vectorizer + Naïve Bayes.

**Text Representations:**
1.  **Count Vectorizer (CV):** Bag-of-Words (BoW) model using unigrams and bigrams.
2.  **BanglaBERT:** A native Bengali transformer model, using the 768-dim `[CLS]` token embedding.
3.  **mBERT:** A multilingual transformer model, also using the `[CLS]` token embedding.

**Classifiers:**
1.  **Logistic Regression (LR)** 
2.  **Support Vector Classifier (SVC)** 
3.  **Multinomial Naive Bayes (MNB)** (Used with CV as a baseline) 

The experimental results for each combination are organized into the following directories:

* `BanglaBERT+LR/`
* `BanglaBERT+SVC/`
* `CV+LR/`
* `CV+SVC/`
* `mBERT+LR/`
* `mBERT+SVC/`
* `MNB/` (Baseline)

## Folder Structure

The repository is organized as follows:

```
├── BanglaBERT+LR/ - Experiments with BanglaBERT and Logistic Regression.
├── BanglaBERT+SVC/ - Experiments with BanglaBERT and Support Vector Classifier.
├── CV+LR/ - Experiments with Count Vectorizer and Logistic Regression.
├── CV+SVC/ - Experiments with Count Vectorizer and Support Vector Classifier.
├── eGov-Lens/ - Individual datasets for each government service.
├── Final Result/ - Consolidated results from all experiments.
├── mBERT+LR/ - Experiments with mBERT and Logistic Regression.
├── mBERT+SVC/ - Experiments with mBERT and Support Vector Classifier.
├── Merged Dataset/ - The merged dataset used for training and evaluation.
├── MNB/ - Experiments with Multinomial Naive Bayes.
├── Topsis_Entropy/ - Model ranking using TOPSIS and Entropy methods.
├── CITATION.cff
├── LICENSE
└── README.md
```

## Results

The `Final Result` directory contains the consolidated performance metrics for each classification task. The `Topsis_Entropy` directory contains the code and results for ranking the different models using the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method with entropy-based weights.

## Cite

If you use this repo, please cite:

```bibtex
@misc{eGovLensCode2025,
  author       = {Fuad, Md. Muqtadir and Mazid, Abdullah Al and Adnan, Md. Istiak},
  title        = {eGov-Lens: Code and Dataset},
  year         = {2025},
  howpublished = {\url{https://github.com/md-muqtadir-fuad/eGov-Lens}},
  note         = {Accessed: 2025-08-25}
}
```