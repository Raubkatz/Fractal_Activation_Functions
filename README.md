# 🧩 Fractal-Activation Playground


This repository explores the use of **fractal activation functions** in shallow feed-forward neural networks, with a focus on tabular classification tasks. In particular, it addresses the following questions:

1. **Do fractal activations improve performance on real tabular benchmarks?**  
   → `01_run_experiments.py`

2. **How stable and reproducible are the results across different random splits and seeds?**  
   → `02_evaluate_results.py`, `02_eval_accuracies_across_Runs.py`

3. **How do fractal activations affect geometric expressivity compared to ReLU and tanh?**  
   → `03_expressivity_experiment_enhanced.py`, `03_expressivity.py`

4. **What do the raw activation functions look like?**  
   → `04_plot_activation_functions.py`, `04_plot_activations.py`

5. **What is the computational cost (training and inference time) of fractal activations?**  
   → `05_analyze_time.py`

6. **How do gradients behave during training when using fractal activations?**  
   → `06_grad_analysis.py`

---

## Repository layout

├── fractal_activation_functions.py

│ # Implementation of all developed fractal activation functions

│

├── 01_run_experiments.py

│ # Main experiment runner: 10 datasets × 5 optimizers × multiple activations

│

├── 02_evaluate_results.py

│ # Aggregates JSON logs into mean / std / min / max performance tables

│

├── 02_eval_accuracies_across_Runs.py

│ # Additional aggregation and comparison of accuracies across random runs

│

├── 03_expressivity_experiment_enhanced.py

│ # Trajectory-length analysis following Poole (2016) and Raghu (2017)

│

├── 03_expressivity.py

│ # Simplified / focused expressivity experiments and visualizations

│

├── 04_plot_activation_functions.py

│ # Plots all activation functions on a fixed input interval (e.g. [-2, 2])

│

├── 04_plot_activations.py

│ # Publication-ready activation plots with consistent styling

│

├── 05_analyze_time.py

│ # Runtime analysis: training time, prediction time, and aggregate statistics

│

├── 06_grad_analysis.py

│ # Per-epoch gradient statistics and stability analysis on a fixed probe batch

│

└── README.md
You are here

## Requirements

python==3.8.19          

numpy==1.23.5           

scipy==1.10.1           

pandas==2.0.3           

scikit-learn==1.3.2     

tensorflow==2.10.0      

matplotlib==3.7.5       

seaborn==0.13.2 


---

## Notes

- All experiments are designed for **small to medium-sized tabular datasets** and **shallow neural networks**.
- Fractal activations are implemented via **finite truncations**, ensuring compatibility with automatic differentiation.
- Runtime and gradient analyses are separated into dedicated scripts to keep the main experiments reproducible and modular.
- Results reported in the accompanying paper are based on **multiple random seeds and train–test splits** to assess robustness.

This repository is intended as a research playground and reference implementation accompanying the manuscript on fractal activation functions.


