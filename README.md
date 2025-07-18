# Fairness_Practices_Robustness_Testing

This paper presents a causal framework for evaluating the robustness of machine learning fairness practices under distributional shifts. By generating neighboring datasets from causal models inferred via discovery algorithms, the approach tests whether common fairness interventions—such as dropping features, hyperparameter tuning, or bias mitigation—maintain fairness across variations in the data. Unlike prior work focused on fairness detection or optimization, our method systematically stress-tests fairness practices to identify those that are sensitive to changes in causal structure or data distribution. The framework is supported by an available tool and validated across six benchmark datasets, demonstrating that many fairness practices fail to generalize robustly.

# Structure of the repository

[Adult_Analysis](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/tree/main/Adult_Analysis): Keeps the results of the causal discovery, probablistic programming, and the RQ1-4 experiments for the Adult dataset.

[Bank_Analysis](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/tree/main/Bank_Analysis): Keeps the results of the causal discovery, probablistic programming, and the RQ1-4 experiments for the Bank dataset.

[Compas_Analysis](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/tree/main/Compas_Analysis): Keeps the results of the causal discovery, probablistic programming, and the RQ1-4 experiments for the Compas dataset.

[Heart_Analysis](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/tree/main/Hear_Analysis): Keeps the results of the causal discovery, probablistic programming, and the RQ1-4 experiments for the Heart dataset.

[Law_Analysis](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/tree/main/Law_Analysis): Keeps the results of the causal discovery, probablistic programming, and the RQ1-4 experiments for the Law dataset.

[Student_Analysis](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/tree/main/Student_Analysis): Keeps the results of the causal discovery, probablistic programming, and the RQ1-4 experiments for the Students dataset.
# Requirments
Python Requirments:

matplotlib==3.10.3

scikit_learn==1.7.0

tensorflow_probability==0.18.0

pandas==2.3.1

mahalanobis==1.2.0

tensorflow==2.10.0

aif360==0.6.1

shap==0.41.0

fairlearn==0.7.0

R Requirments:

pcalg==2.7-12

rstan==2.32.7

Rgraphviz==2.52.0


#  Causal Discovery

Our approach employs three causal discovery algorithms—PC, SIMY, and GES—to infer the causal graphs of the datasets. This represents the first step of our methodology, as the outputs of these experiments are subsequently used to address our research questions. We implemented the causal discovery process in R using the pcalg package, as detailed in the [Causal_discovery.R](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/Causal_Discovery.R) script. This script infers the causal graphs for the Bank dataset using all three algorithms. To apply the procedure to other datasets, the corresponding dataset path must be uncommented in the code (e.g., for the Adult dataset, uncomment the line data_path <- './subjects/datasets/adult_org-Copy1.csv'). The output is a CSV file representing the Completed Partially Directed Acyclic Graph (CPDAG) for each dataset. The script stores the CPDAGs in the directory ./{Dataset}_Analysis/{DiscoveryAlgorithm}/DAGs/ using the naming format {Dataset}_{DiscoveryAlgorithm}.csv. For instance, the CPDAG for the Bank dataset using the PC algorithm is saved as ./Bank_Analysis/pc/DAGs/Bank_pc.csv. Note that the resulting CPDAG may still contain some bidirectional edges due to equivalence class ambiguity. 
```
cd Causal_Discovery
Rscript Causal_discovery.R

```
The next step is to prepare the possible Directed Acyclic Graphs (DAGs) from the CPDAG result for each discovery algorithm. The Python script [Preparing_DAGs.py](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/Preparing_DAGs.py) reads the CPDAG corresponding to each dataset and causal discovery algorithm and generates all compatible DAGs. The output of this script is saved to ./{Dataset}_Analysis/{DiscoveryAlgorithm}/DAGs/ using the format {Dataset}_{DiscoveryAlgorithm}_{DAG}_{DAG number}.csv. Bellow command runs the code for Bank dataset using PC algorithm.

```
cd Causal_Discovery
python Preparing_DAGs.py --dataset Bank --alg pc

```

# Inferring the Weights of DAGs

Having obtained the DAGs for all datasets and discovery algorithms, we then use probabilistic programming to infer the posterior distribution of the weights associated with each edge in the DAGs. To accomplish this, we use RSTAN. A Python script, [Py_2_R.py](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/Py_2_R.py), takes a DAG as input and automatically generates the corresponding RStan probabilistic code. Bellow command runs this script for Bank dataset using simy algorithm.
```
cd Probablistic_Programming
python Py_2_R.py --dataset Bank --alg simy

```
The generated .stan code is saved in the directory ./{Dataset}_Analysis/{DiscoveryAlgorithm}/PP/ using the naming convention {Dataset}_{DiscoveryAlgorithm}_{DAG}_{DAG number}.stan. Each generated STAN model is then executed using a dedicated R script located at ./{Dataset}_Analysis/Rstan_shell_{DAG number}.R, which runs the RStan code for the corresponding dataset, discovery algorithm, and DAG number. Example bellow runs the Rstan file to infer the weights of the causal graph 58 of simy algorithm.
```
cd Bank_Analysis
Rscript Rstan_simy_58.R

```

# Fairness Practices Robustness Testing Tool
The python file [Robustness_Test.py](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/Robustness_Test.py) implements of our Fairness Practices Robusness Testing. To run the tool for the selected dataset (Adult, Bank, Compas, Law, Heart, and Student) and selcted practices ('SelectKBest', 'SelectFpr','SelectPercentile' ,'drop', 'TO', 'CEO', and 'HP'), argument --dataset selects the dataset to test and --practice selects the fairness practices to do robusteness test. Below comand runs the tool for Adult dataset and SelectKBest practice.


```
python Robustness_Test.py --dataset Adult --practice SelectKBest

```

# RQ1

In this experiment, we evaluate the quality of the generated data produced by each causal model to eliminate discovery algorithms that demonstrate a lower success rate compared to others. The Python script [RQ1.py](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/RQ1.py) implements this evaluation. It leverages the inferred posterior weights for each DAG and dataset, which are saved in the format ./{Dataset}_Analysis/{DiscoveryAlgorithm}/PP/{Dataset}_{DiscoveryAlgorithm}_PP_{DAG number}.csv. The outcomes of this quality assessment, as reported in Table 1, are stored in ./{Dataset}_Analysis/RQ1/{Dataset}_{DiscoveryAlgorithm}_RQ1_results.npy. To benchmark the performance, we include two baseline models: one based on the corresponding DAGs with equal edge weights and another using randomly generated samples. These baselines are implemented in the Python script [RND_EQ.py](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/RND_EQ.py). The results for the random sampling baseline are saved to ./{Dataset}_Analysis/RQ1_RND/{Dataset}_{DiscoveryAlgorithm}_RQ1_results_RND.npy, while the equal-weights baseline results are saved to ./{Dataset}_Analysis/RQ1_RND/{Dataset}_{DiscoveryAlgorithm}_RQ1_results_eq.npy.
```
cd RQs
python RQ1.py --dataset Adult
```
# RQ2
In this experiment, we study the robustness of in-processing fairness practices when applied to causally generated data. Specifically, we assess the sensitivity of fairness outcomes to the removal of sensitive and non-sensitive features. For non-sensitive feature removal, we evaluate three common selection methods: SelectKBest, SelectFpr, and SelectPercentile. The experiment is implemented in the Python script [RQ2.py](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/RQ2.py), and the results—reported in Table 3—are saved in the format ./{Dataset}_Analysis/RQ2/{DiscoveryAlgorithm}_{Fairness Practice}_EOD_{DAG number}.npy. This command runs RQ2 experiments for Adult dataset.
```
cd RQs
python RQ2.py --dataset Adult
```
 
To further understand the value of incorporating causal graphs, we conduct an ablation study where the same fairness evaluation is performed without using causal graphs. This baseline scenario is implemented in the script [RQ2_Ablation.py](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/RQ2_Ablation.py), and the corresponding results are reported in Table 4. The bellow command runs the ablation study for Bank dataset.

```
cd RQs
python RQ2_Ablation.py --dataset Adult
```

# RQ3
We conduct a series of experiments tounderstand if some hyperparameters (HPs) can systematically influence fairness.We adopted an in-processing fairness mitigation tool from Tizpaz-Niari et al. [6], which is designed to explore the hyperparameter space of ML models using genetic algorithms to find configurations that yield fairer outcomes for a given dataset. This tool applies mutation operators to evolve candidate configurations and identify ones that minimize fairness violations according to a predefined metric [RQ#.py](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/RQ3.py) replicates the results reported in Table 6.  The results of this script is saved to ./{Dataset}_Analysis/RQ2/{dataset}_RQ3.npy. 
```
cd RQs
python RQ3.py --dataset Adult
```

# RQ4
We examine two well-established post-processing bias mitigation algorithms: Threshold Optimizer and Calibrated Equalized Odds. Our primary objective in this
experiment is to analyze the robustness of these bias mitigation algorithms across different datasets. The python Script [RQ4.py](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/RQs/RQ4.py). The bellow command runs the robustness test for Threshold Optimizer (TO) on Adult dataset.

```
cd RQs
python RQ4.py --dataset Adult --mitigator TO
```
# Utility Functions

Here we explain the functions we used to generate the overview plots and tables.

[Overview_plots.ipynb](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/Utils_Function/Overview_plots.ipynb) uses the results to generates the plots used in the paper.

[RQ1&2_Results.ipynb](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/Utils_Function/RQ1%262_Results.ipynb) uses the results of RQ1 and RQ2 to generate LATEX code for the results reported in Table 2&3.

[RQ3_results.ipynb](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/Utils_Function/RQ3_results.ipynb) uses the results of RQ3 to generate LATEX code for the results reported in Table 5.

[RQ4_results.ipynb](https://github.com/armanunix/Fairness_Practices_Robustness_Testing/blob/main/Utils_Function/RQ4_Results.ipynb) analyzes the results of RQ4 to generate LATEX code for the results reported in Table 7.
