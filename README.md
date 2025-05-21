# 🧩 Fractal-Activation Playground
A fully-reproducible sandbox accompanying  
**“Fractals in Neural Networks: Introducing a New Class of Activation Functions.”**

The repo answers three questions:

1. **Do fractal activations help on real tabular benchmarks?** → `01_run_experiments.py`  
2. **How stable / accurate are they across 40 random splits?** → `02_evaluate_results.py`  
3. **Why do they possess higher *expressivity* than ReLU/Tanh?** → `03_expressivity_experiment_enhanced.py`  
4. **What do the raw curves look like?** → `04_plot_activation_functions.py`  

---

## 💾 Repository layout

fractal-activation-playground/

├── fractal_activation_functions.py # all 12 custom σ(·)

│

├── 01_run_experiments.py # 🚂 grid-search: 10 data sets × 5 optims × 12 activations

├── 02_evaluate_results.py # 📊 merge JSON logs → mean / std / min / max tables

├── 03_expressivity_experiment_enhanced.py

│ # 🔍 replicates Poole-16 / Raghu-17 trajectory analysis

├── 04_plot_activation_functions.py # 🎨 pretty plots of every activation on [-2,2]

│

├── requirements.txt # “pip install -r …” yields TF + SciPy stack

└── README.md # you are here

## Requirements

### ── core language ────────────────────────────────────────────

python==3.8.19          # same interpreter version used in the conda env

### ── numerical stack ─────────────────────────────────────────

numpy==1.23.5           # base ndarray operations everywhere

scipy==1.10.1           # required by scikit-learn, not imported directly

### ── data wrangling / ML utilities ───────────────────────────

pandas==2.0.3           # result aggregation in 02_evaluate_results.py

scikit-learn==1.3.2     # metrics, preprocessing, model_selection

### ── deep-learning backend ───────────────────────────────────

tensorflow==2.10.0      # all models + fractal activations

### ── visualisation ───────────────────────────────────────────

matplotlib==3.7.5       # loss curves, expressivity plots, activation curves

seaborn==0.13.2         # prettier styling for expressivity & activation figures

