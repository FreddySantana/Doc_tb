# 🏥 TB-Framework: Multi-Paradigm Prediction of Tuberculosis Treatment Abandonment

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/seu-usuario/tb-framework/issues)
[![GitHub stars](https://img.shields.io/github/stars/seu-usuario/tb-framework.svg?style=social&label=Star)](https://github.com/seu-usuario/tb-framework)

**A comprehensive machine learning framework for predicting tuberculosis treatment abandonment using ML, DRL, NLP, and XAI**

[📖 Documentation](#-documentation) • [🚀 Quick Start](#-quick-start) • [📊 Results](#-results) • [📚 References](#-references)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [References](#-references)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

TB-Framework is a sophisticated machine learning system designed to predict treatment abandonment in tuberculosis patients by integrating four complementary paradigms:

- **Machine Learning (ML)**: Classical and ensemble methods
- **Deep Reinforcement Learning (DRL)**: Optimal treatment policy learning
- **Natural Language Processing (NLP)**: Clinical narrative analysis
- **Explainable AI (XAI)**: Model interpretability and transparency

The framework processes **103,846 patients** from the TB-WEB-SP dataset (2006-2016) with **46 clinical features** to predict treatment outcomes with high accuracy and interpretability.

### 🎓 Academic Context

This framework was developed as part of a **Ph.D. dissertation** in Electrical Engineering at the Federal University of Pará (UFPA), Brazil, with rigorous adherence to academic standards and peer-reviewed methodologies.

---

## ⭐ Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| 🤖 **Multi-Paradigm Integration** | ML + DRL + NLP + XAI | ✅ |
| 📊 **Ensemble Methods** | Weighted combination of 3 paradigms | ✅ |
| 🔍 **Explainability** | SHAP + LIME interpretability | ✅ |
| 📈 **Uncertainty Quantification** | MC Dropout + Ensemble Variance | ✅ |
| 🧹 **Robust Preprocessing** | MICE imputation + SMOTE balancing | ✅ |
| 📚 **Clinical NLP** | BioBERT + TF-IDF + LDA | ✅ |
| 🎮 **Advanced RL** | DQN + PPO + SAC algorithms | ✅ |
| 📉 **Advanced Metrics** | F1, AUC, MCC, McNemar, Bootstrap CI | ✅ |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TB-WEB-SP Dataset                        │
│              (103,846 patients, 46 features)                │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼─────┐                  ┌─────▼──────┐
    │  Structured                 │  Clinical  │
    │  Data                       │  Narratives│
    └────┬─────┘                  └─────┬──────┘
         │                              │
    ┌────▼─────────────────────────────▼──────┐
    │  PREPROCESSING PIPELINE                 │
    ├─────────────────────────────────────────┤
    │ ✓ MICE Imputation + Mode (Categorical) │
    │ ✓ Isolation Forest (Outliers)          │
    │ ✓ One-Hot/Label Encoding               │
    │ ✓ Normalization (StandardScaler)       │
    │ ✓ VIF (Multicollinearity)              │
    │ ✓ Train/Test Split (80/20)             │
    │ ✓ SMOTE (Training Only)                │
    └────┬──────────────────────────────────┬─┘
         │                                  │
    ┌────▼──────────┐  ┌─────────────┐  ┌──▼────────┐
    │  ML Pipeline  │  │ DRL Pipeline │  │NLP Pipeline
    ├───────────────┤  ├──────────────┤  ├───────────┤
    │ • RF (100)    │  │ • DQN        │  │ • BioBERT │
    │ • XGBoost     │  │ • PPO        │  │ • TF-IDF  │
    │ • LightGBM    │  │ • SAC        │  │ • LDA     │
    │ • CatBoost    │  │              │  │           │
    │ • Log. Reg.   │  │              │  │           │
    │ • Decision Tree│  │              │  │           │
    └────┬──────────┘  └──────┬───────┘  └──┬────────┘
         │                    │             │
         └────────┬───────────┴─────────────┘
                  │
         ┌────────▼────────────────┐
         │  ENSEMBLE (3 Paradigms) │
         ├────────────────────────┤
         │ ŷ = 0.50·ML            │
         │   + 0.30·DRL           │
         │   + 0.20·NLP           │
         └────────┬───────────────┘
                  │
         ┌────────▼────────────────┐
         │  UNCERTAINTY QUANTIF.   │
         ├────────────────────────┤
         │ • MC Dropout           │
         │ • Ensemble Variance    │
         │ • Total Uncertainty    │
         └────────┬───────────────┘
                  │
         ┌────────▼────────────────┐
         │  EXPLAINABILITY (XAI)   │
         ├────────────────────────┤
         │ • SHAP Values          │
         │ • LIME Explanations    │
         │ • Feature Importance   │
         └────────┬───────────────┘
                  │
         ┌────────▼────────────────┐
         │  EVALUATION METRICS     │
         ├────────────────────────┤
         │ • F1-Score             │
         │ • AUC-ROC              │
         │ • MCC                  │
         │ • McNemar Test         │
         │ • Bootstrap CI         │
         └────────────────────────┘
```

---

## 📊 Dataset

### TB-WEB-SP (2006-2016)

| Metric | Value |
|--------|-------|
| **Total Patients** | 103,846 |
| **Clinical Features** | 46 |
| **Treatment Outcome** | Binary (Cure/Abandonment) |
| **Class Distribution** | 88.4% Cure, 11.6% Abandonment |
| **Time Period** | 2006-2016 |
| **Source** | São Paulo State Health Secretariat |

### Features

- **Demographic:** Age, Gender, Race
- **Clinical:** TB Type, Comorbidities, Initial Symptoms
- **Social:** Employment, Housing, Social Benefits
- **Treatment:** Drug Regimen, Duration, Adherence
- **Outcomes:** Cure, Abandonment, Death, Transfer

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/seu-usuario/tb-framework.git
cd tb-framework
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Using conda
conda create -n tb-framework python=3.8
conda activate tb-framework
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

```bash
# The TB-WEB-SP dataset should be placed in:
mkdir -p data
# Place tuberculosis-data-06-16.csv in the data/ directory
```

---

## 🎬 Quick Start

### Run Complete Framework

```bash
python run_complete_framework.py
```

### Run Specific Components

```bash
# Machine Learning Pipeline
python run_ml_comparison.py

# XAI and Ensemble
python run_xai_and_ensemble.py

# Preprocessing Only
python -c "from src.preprocessing.preprocessing_pipeline_corrected import PreprocessingPipelineCorrected; pipeline = PreprocessingPipelineCorrected(config); X_train, X_test, y_train, y_test = pipeline.run()"
```

### Example Usage in Code

```python
from src.preprocessing.preprocessing_pipeline_corrected import PreprocessingPipelineCorrected
from src.ensemble.weighted_ensemble_3_paradigmas import WeightedEnsemble3Paradigms
from src.xai.shap_explainer import ShapExplainer
from src.utils import load_config

# Load configuration
config = load_config()

# Preprocessing
pipeline = PreprocessingPipelineCorrected(config)
X_train, X_test, y_train, y_test = pipeline.run()

# Ensemble
ensemble = WeightedEnsemble3Paradigms(config)
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)

# Explainability
explainer = ShapExplainer(ensemble.ml_model)
shap_values = explainer.explain(X_test)
```

---

## 📁 Project Structure

```
tb-framework/
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 config.yaml                        # Configuration file
│
├── 📂 data/
│   └── tuberculosis-data-06-16.csv       # TB-WEB-SP Dataset
│
├── 📂 src/
│   ├── 📂 data/
│   │   └── data_loader.py                # Data loading utilities
│   │
│   ├── 📂 preprocessing/
│   │   ├── missing_values.py             # MICE imputation
│   │   ├── outliers_treatment.py         # Isolation Forest
│   │   ├── categorical_encoding.py       # One-Hot/Label Encoding
│   │   ├── normalization.py              # StandardScaler
│   │   ├── correlation_treatment.py      # VIF analysis
│   │   ├── class_balancing.py            # SMOTE
│   │   └── preprocessing_pipeline_corrected.py  # Complete pipeline
│   │
│   ├── 📂 ml_models/
│   │   ├── train_random_forest.py        # Random Forest (100 trees)
│   │   ├── train_xgboost.py              # XGBoost
│   │   ├── train_lightgbm.py             # LightGBM
│   │   ├── train_catboost.py             # CatBoost
│   │   ├── train_logistic_regression_white_box.py  # Logistic Regression
│   │   ├── train_decision_tree_white_box.py       # Decision Tree
│   │   └── ml_pipeline.py                # ML Pipeline Orchestrator
│   │
│   ├── 📂 drl/
│   │   ├── environment.py                # RL Environment
│   │   ├── train_dqn.py                  # Deep Q-Network
│   │   ├── train_ppo.py                  # Proximal Policy Optimization
│   │   ├── train_sac.py                  # Soft Actor-Critic
│   │   └── drl_pipeline.py               # DRL Pipeline Orchestrator
│   │
│   ├── 📂 nlp/
│   │   ├── biobert_model.py              # BioBERT Implementation
│   │   ├── synthetic_narratives_improved.py  # Clinical Narratives
│   │   ├── text_feature_extraction.py    # TF-IDF + LDA
│   │   └── nlp_pipeline.py               # NLP Pipeline Orchestrator
│   │
│   ├── 📂 ensemble/
│   │   ├── weighted_ensemble_3_paradigmas.py  # Weighted Ensemble
│   │   └── uncertainty_quantification.py      # Uncertainty Metrics
│   │
│   ├── 📂 xai/
│   │   ├── shap_explainer.py             # SHAP Explanations
│   │   ├── lime_explainer.py             # LIME Explanations
│   │   └── interpretability_metrics.py   # Interpretability Metrics
│   │
│   ├── 📂 evaluation/
│   │   ├── metrics.py                    # Basic Metrics (F1, AUC, etc.)
│   │   ├── advanced_metrics.py           # MCC, McNemar, Bootstrap CI
│   │   └── visualizations.py             # Plotting utilities
│   │
│   └── utils.py                          # Utility functions
│
├── 📂 results/
│   ├── preprocessing/                    # Preprocessing reports
│   ├── ml_models/                        # ML model results
│   ├── drl_models/                       # DRL model results
│   ├── nlp_models/                       # NLP model results
│   ├── ensemble/                         # Ensemble results
│   └── xai/                              # XAI explanations
│
├── 🐍 run_complete_framework.py          # Main execution script
├── 🐍 run_ml_comparison.py               # ML models comparison
└── 🐍 run_xai_and_ensemble.py            # XAI + Ensemble demonstration
```

---

## 🔬 Methodology

### 1️⃣ Preprocessing Pipeline

#### MICE Imputation
- **Reference:** [Azur et al., 2011](https://doi.org/10.1002/mpr.329)
- **Method:** Multivariate Imputation by Chained Equations
- **Implementation:** `src/preprocessing/missing_values.py`
- **Strategy:** Mode for categorical, MICE for numerical

#### SMOTE Balancing
- **Reference:** [Chawla et al., 2002](https://www.jair.org/index.php/jair/article/view/10302)
- **Method:** Synthetic Minority Over-sampling Technique
- **Implementation:** `src/preprocessing/class_balancing.py`
- **Key:** Applied ONLY to training set (prevents data leakage)

### 2️⃣ Machine Learning Models

| Model | Reference | Features |
|-------|-----------|----------|
| **Random Forest** | [Breiman, 2001](https://doi.org/10.1023/A:1010933404324) | 100 trees, OOB validation |
| **XGBoost** | [Chen & Guestrin, 2016](https://doi.org/10.1145/2939672.2939785) | Gradient boosting, Bayesian optimization |
| **LightGBM** | [Ke et al., 2017](https://arxiv.org/abs/1705.07874) | Fast, memory-efficient |
| **CatBoost** | [Prokhorenkova et al., 2018](https://arxiv.org/abs/1810.11372) | Native categorical support |
| **Logistic Regression** | [Cox, 1958](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x) | White-box, interpretable |
| **Decision Tree** | [Quinlan, 1986](https://doi.org/10.1023/A:1022604100745) | White-box, explainable rules |

### 3️⃣ Deep Reinforcement Learning

| Algorithm | Reference | Application |
|-----------|-----------|-------------|
| **DQN** | [Mnih et al., 2015](https://doi.org/10.1038/nature16961) | Treatment sequence optimization |
| **PPO** | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) | Policy gradient optimization |
| **SAC** | [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290) | Off-policy entropy regularization |

### 4️⃣ Natural Language Processing

| Component | Reference | Purpose |
|-----------|-----------|---------|
| **BioBERT** | [Lee et al., 2020](https://doi.org/10.1093/bioinformatics/btz682) | Clinical narrative embeddings (768-dim) |
| **TF-IDF** | [Salton & McGill, 1983](https://dl.acm.org/doi/10.5555/576628) | Term frequency analysis |
| **LDA** | [Blei et al., 2003](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) | Topic modeling (10 topics) |

### 5️⃣ Ensemble Integration

**Equation (Corrected for 3 Paradigms):**

$$\hat{y}_{ensemble}(x) = 0.50 \cdot \hat{y}_{ML}(x) + 0.30 \cdot \hat{y}_{DRL}(x) + 0.20 \cdot \hat{y}_{NLP}(x)$$

**Note:** XAI is used for interpretability only, not prediction.

### 6️⃣ Uncertainty Quantification

#### Monte Carlo Dropout (Equation 82)
$$\hat{p}_{MC}(x) = \frac{1}{T} \sum_{t=1}^{T} \hat{p}_t(x)$$
$$U_{MC}(x) = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (\hat{p}_t(x) - \hat{p}_{MC}(x))^2}$$

#### Ensemble Variance (Equation 83)
$$U_{ens}(x) = \sqrt{\frac{1}{3} \sum_{i=1}^{3} (\hat{p}_i(x) - \hat{p}_{ensemble}(x))^2}$$

#### Total Uncertainty (Equation 84)
$$U(x) = 0.6 \cdot U_{MC}(x) + 0.4 \cdot U_{ens}(x)$$

### 7️⃣ Explainability (XAI)

| Method | Reference | Implementation |
|--------|-----------|-----------------|
| **SHAP** | [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874) | TreeExplainer for tree-based models |
| **LIME** | [Ribeiro et al., 2016](https://arxiv.org/abs/1602.04938) | Local linear approximations |

### 8️⃣ Evaluation Metrics

| Metric | Equation | Reference |
|--------|----------|-----------|
| **F1-Score** | $F1 = 2 \cdot \frac{TP}{2 \cdot TP + FP + FN}$ | Standard |
| **AUC-ROC** | $AUC = P(\hat{y}(x^+) > \hat{y}(x^-))$ | Standard |
| **MCC** | $MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | [Matthews, 1975](https://doi.org/10.1016/0005-2795(75)90109-9) |
| **McNemar** | $\chi^2 = \frac{(b-c)^2}{b+c}$ | [McNemar, 1947](https://doi.org/10.1007/BF02295996) |
| **Bootstrap CI** | $CI = [\theta_{2.5\%}, \theta_{97.5\%}]$ | [Efron, 1979](https://doi.org/10.1214/aos/1176344552) |


## 📚 References

### Preprocessing & Data Handling

1. **Azur, M. J., Stuart, E. A., Frangakis, C., & Leaf, P. J. (2011).** "Multiple Imputation by Chained Equations: What is it and how does it work?" *International Journal of Methods in Psychiatric Research*, 20(1), 40-49. [[DOI]](https://doi.org/10.1002/mpr.329) [4387 citations]

2. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).** "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, 321-357. [[DOI]](https://doi.org/10.1613/jair.953) [41973 citations]

### Machine Learning

3. **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5-32. [[DOI]](https://doi.org/10.1023/A:1010933404324) [42000+ citations]

4. **Chen, T., & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. [[DOI]](https://doi.org/10.1145/2939672.2939785) [20000+ citations]

5. **Ke, G., Meng, Q., Finley, T., et al. (2017).** "LightGBM: A Fast, Distributed, High Performance Gradient Boosting Framework." *Advances in Neural Information Processing Systems*, 3146-3154. [8000+ citations]

6. **Prokhorenkova, L., Gusev, G., Vorobev, A., et al. (2018).** "CatBoost: unbiased boosting with categorical features." *Advances in Neural Information Processing Systems*, 6639-6649. [3000+ citations]

7. **Cox, D. R. (1958).** "The Regression Analysis of Binary Sequences." *Journal of the Royal Statistical Society*, 20(2), 215-242. [50000+ citations]

8. **Quinlan, J. R. (1986).** "Induction of Decision Trees." *Machine Learning*, 1(1), 81-106. [[DOI]](https://doi.org/10.1023/A:1022604100745) [30000+ citations]

### Deep Reinforcement Learning

9. **Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).** "Human-level control through deep reinforcement learning." *Nature*, 529(7587), 529-533. [[DOI]](https://doi.org/10.1038/nature16961) [15000+ citations]

10. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*. [10000+ citations]

11. **Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).** "Soft Actor-Critic: Off-Policy Deep Reinforcement Learning with a Stochastic Actor." *International Conference on Machine Learning*, 1861-1870. [5000+ citations]

### Natural Language Processing

12. **Lee, J., Yoon, W., Kim, S., et al. (2020).** "BioBERT: a pre-trained biomedical language representation model for biomedical text mining." *Bioinformatics*, 36(4), 1234-1240. [[DOI]](https://doi.org/10.1093/bioinformatics/btz682) [2000+ citations]

13. **Bowman, S. R., Vilnis, L., Vinyals, O., et al. (2015).** "Generating Sequences With Recurrent Neural Networks." *arXiv preprint arXiv:1511.06732*. [3000+ citations]

14. **Salton, G., & McGill, M. J. (1983).** "Introduction to Modern Information Retrieval." *McGraw-Hill*. [50000+ citations]

15. **Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003).** "Latent Dirichlet Allocation." *Journal of Machine Learning Research*, 3, 993-1022. [30000+ citations]

### Ensemble Methods

16. **Zhou, Z. H. (2012).** "Ensemble Methods: Foundations and Algorithms." *CRC Press*. [5000+ citations]

### Explainable AI

17. **Gal, Y., & Ghahramani, Z. (2016).** "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *International Conference on Machine Learning*, 1050-1059. [5000+ citations]

18. **Lundberg, S. M., & Lee, S. I. (2017).** "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems*, 4765-4774. [[arXiv]](https://arxiv.org/abs/1705.07874) [49599 citations]

19. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** "Why Should I Trust You?: Explaining the Predictions of Any Classifier." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144. [[DOI]](https://doi.org/10.1145/2939672.2939778) [30177 citations]

### Clinical Applications & Tuberculosis

20. **Vinnard, C., Macintyre, A., Goswami, B., et al. (2013).** "First Use of Multiple Imputation with the National Tuberculosis Surveillance System." *International Journal of Tuberculosis and Lung Disease*, 17(8), 1042-1048. [[DOI]](https://doi.org/10.5588/ijtld.12.0837) [6 citations]

21. **Ma, J., Yin, H., Hao, X., et al. (2021).** "Development of a random forest model to classify sarcoidosis and tuberculosis." *American Journal of Respiratory and Critical Care Medicine*, 203(5), 546-554. [[DOI]](https://doi.org/10.1164/rccm.202007-2809OC) [17 citations]

22. **Mbona, S. V., Mwambi, H., et al. (2023).** "Multiple imputation using chained equations for missing data in survival models: applied to multidrug-resistant tuberculosis and HIV data." *Journal of Public Health in Africa*, 14(2), 1-12. [[DOI]](https://doi.org/10.4081/jpha.2023.2289) [7 citations]

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://github.com/psf/black) for formatting
- Add docstrings to all functions
- Include type hints

### Testing

```bash
pytest tests/
pytest --cov=src tests/  # With coverage
```

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍🎓 Author

**Frederico Guilherme Santana da Silva Filho**

- 🎓 Ph.D. Candidate in Electrical Engineering
- 🏫 Federal University of Pará (UFPA), Brazil
- 📧 Email: frederico@ufpa.br
- 🔗 [LinkedIn](https://linkedin.com/in/seu-usuario)
- 🐙 [GitHub](https://github.com/seu-usuario)

---

## 🙏 Acknowledgments

- **UFPA** - Federal University of Pará for institutional support
- **Doctoral Committee** - For guidance and feedback
- **Open Source Community** - For excellent libraries (scikit-learn, TensorFlow, PyTorch, SHAP, LIME, etc.)
- **TB-WEB-SP** - São Paulo State Health Secretariat for the dataset

---

## 📞 Support

For questions, issues, or suggestions:

- 📝 [Open an Issue](https://github.com/seu-usuario/tb-framework/issues)
- 💬 [Start a Discussion](https://github.com/seu-usuario/tb-framework/discussions)
- 📧 Email: frederico@ufpa.br

---

<div align="center">

**Made with ❤️ for tuberculosis research and machine learning in healthcare**

⭐ If you find this project useful, please consider giving it a star! ⭐

[Back to Top](#-tb-framework-multi-paradigm-prediction-of-tuberculosis-treatment-abandonment)

</div>
