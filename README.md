# 🧩 Fractal-Activation Playground

This repository dsicusses the following:

1. **Do fractal activations help on real tabular benchmarks?** → `01_run_experiments.py`  
2. **How stable / accurate are they across different random splits?** → `02_evaluate_results.py`  
3. **Why do they possess higher *expressivity* than ReLU/Tanh?** → `03_expressivity_experiment_enhanced.py`  
4. **What do the raw curves look like?** → `04_plot_activation_functions.py`  

---

## Repository layout

fractal-activation-playground/

├── fractal_activation_functions.py # all eveloped fractal activaiton fucntions

├── 01_run_experiments.py # 10 data sets × 5 optims × 12 activations

├── 02_evaluate_results.py # merge JSON logs → mean / std / min / max tables

├── 03_expressivity_experiment_enhanced.py # replicates Poole-16 / Raghu-17 trajectory analysis

├── 04_plot_activation_functions.py # pretty plots of every activation on [-2,2]

└── README.md # you are here

## Requirements

python==3.8.19          

numpy==1.23.5           

scipy==1.10.1           

pandas==2.0.3           

scikit-learn==1.3.2     

tensorflow==2.10.0      

matplotlib==3.7.5       

seaborn==0.13.2        

