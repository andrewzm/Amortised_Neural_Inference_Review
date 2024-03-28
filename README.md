# Source code for "Neural Methods for Amortised Statistical Inference"

![Figure 2: Illustration of amortised likelihood-to-evidence ratio estimation](/fig/Bayes_classifier.png?raw=true)

This repository contains code for reproducing the results in the review paper "Neural Methods for Amortised Statistical Inference" [(Zammit-Mangion, Sainsbury-Dale, Huser, 2024+)](). *TODO* Manuscript URL when it is available. 

The code in this repository is made available primarily for reproducibility purposes, and we encourage readers seeking to implement neural methods for amortised statistical inference to explore the cited software packages and their documentation. In particular, the software packages:
1. [NeuralEstimators](https://github.com/msainsburydale/NeuralEstimators.jl) (Julia and R)
1. [sbi](https://github.com/sbi-dev/sbi) (Python)
1. [LAMPE](https://github.com/probabilists/lampe) (Python)
1. [BayesFlow](https://github.com/stefanradev93/BayesFlow) (Python)
1. [swyft](https://github.com/undark-lab/swyft) (Python)

Note that each of these packages can be interfaced from R using, for example, [reticulate](https://rstudio.github.io/reticulate/). 


## Instructions

First, download this repository and navigate to its top-level directory within terminal.

### Software dependencies

*TODO* Any specific instructions for getting reticulate set up properly? 
*TODO* Better way of saving and installing R package dependencies, like I do with the Julia packages? 

Before installing the software dependencies, users may wish to set up a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) environment, so that the dependencies of this repository do not affect the user's current installation. To create a conda environment, run the following command in terminal:

```
conda create -n ARSIA -c conda-forge julia=1.9.4 r-base nlopt
```

Then activate the conda environment with:

```
conda activate ARSIA
```

The above conda environment installs Julia and R automatically. If you do not wish to use a conda environment, you will need to install Julia and R manually if they are not already on your system:  

- Install [Julia 1.9.4](https://julialang.org/downloads/).
- Install [R >= 4.0.0](https://www.r-project.org/).



Once Julia and R are setup, install the Julia and R package dependencies (given in `Project.toml` and `Manifest.toml`, and `dependencies.txt`, respectively) by running the following commands from the top-level of the repository: *TODO* dependencies.txt is not currently in the repo: how do we want to document and install the R package dependencies? 

```
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```
```
Rscript dependencies_install.R
```

### Hardware requirements

In general, the fast training of neural networks requires graphical processing units (GPUs). However, the code in this repository also runs on the central processing unit (CPU) in only a moderate amount of time. Therefore, there are no major hardware requirements for running this code. 

### Reproducing the results

The repository is organised into folders containing source code (`src`), intermediate objects generated from the source code (`output`), and figures (`fig`).

The replication script is `run.sh`, invoked using `bash run.sh` from the top level of this repository. The replication script will automatically train the neural networks, generate estimates/samples from both the neural and likelihood-based estimators/samplers, and populate the `fig` folder with the figures and results of the manuscript.

Note that the nature of our experiments means that the run time for reproducing the results of the manuscript can be moderate (on the order of several hours). 
