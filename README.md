# Source code for "Neural Methods for Amortised Statistical Inference" (Under Construction)

![Figure 2: Illustration of amortised likelihood-to-evidence ratio estimation](/fig/Bayes_classifier.png?raw=true)

This repository contains code for reproducing the results in the review paper "Neural Methods for Amortised Statistical Inference" by Andrew Zammit-Mangion, Matthew Sainsbury-Dale, and Raphaël Huser.

The code in this repository is made available primarily for reproducibility purposes. Readers seeking to implement neural methods for amortised statistical inference should also explore the cited software packages and their documentation. In particular, the software packages:
1. [BayesFlow](https://github.com/stefanradev93/BayesFlow) (Python)
1. [LAMPE](https://github.com/probabilists/lampe) (Python)
1. [NeuralEstimators](https://github.com/msainsburydale/NeuralEstimators.jl) (Julia and R)
1. [sbi](https://github.com/sbi-dev/sbi) (Python)
1. [swyft](https://github.com/undark-lab/swyft) (Python)

Note that each of these packages can be interfaced from R using [reticulate](https://rstudio.github.io/reticulate/). 


### Software dependencies

We suggest that users set up a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) environment, so that the dependencies of this repository do not affect the user's current installation. To create a conda environment, run the following command in terminal (note, we are not installing GPU libraries as these are not needed for this code):

```
conda create -n ARSIA -c conda-forge julia=1.9.4 r-base nlopt bayesflow
```

Then activate the conda environment with:

```
conda activate ARSIA
```

Once this is done, you will need to point the R scripts to the correct conda environment so it uses the correct Python environment -- simply change the argument to the `use_condaenv()` function in `src/1_Generate_GP_Data.R`, `src/3_fKL.R`, `4_rKL.R`, `5_rKL_Synthetic_Naive.R`, and `6_rKL_Synthetic_MutualInf.R` to the path of your conda environment.

If you do not wish to use a conda environment, you will need to install Python, Julia, R and the associated packages manually if they are not already on your system:  

- Install [Julia 1.9.4](https://julialang.org/downloads/).
- Install [R >= 4.0.0](https://www.r-project.org/).
- Install [Python >= 3.10.0](https://www.python.org/).

Once Julia and R are setup, install the Julia and R package dependencies by running the following commands from the top-level of the repository:

```
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```
```
Rscript dependencies_install.R
```

### Hardware requirements

In general, the fast training of neural networks requires GPUs. However, the code in this repository also runs on CPUs in a moderate amount of time. Therefore, there are no major hardware requirements for running this code. 

### Reproducing the results

First, download this repository and navigate to its top-level directory within terminal.

The repository is organised into folders containing source code (`src`), intermediate objects generated from the source code (`output`), and figures (`fig`). Checkpoints that contain pre-trained networks are available (`ckpts`).

The replication script is `run.sh`, invoked using `bash run.sh` from the top level of this repository. The replication script will automatically train the neural networks, generate estimates/samples from both the neural and likelihood-based estimators/samplers, and populate the `fig` folder with the figures and results of the manuscript.

Note that the nature of our experiments means that the run time for reproducing the results of the manuscript can be moderate (on the order of several hours). 
