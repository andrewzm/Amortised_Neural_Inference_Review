# BayesFlow template R script: Estimation of length scale in Gaussian 
# Process covariance function
#
# Author: Andrew Zammit-Mangion, azm (at) uow.edu.au
# Date: 2024-02-15
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

##  Load core R packages  
library(fields)
library(dplyr)

## Load BayesFlow in R Conda -- note, needs to be changed according to your system
library("reticulate")
use_condaenv("~/miniconda3/envs/BayesFlow")
library(tensorflow)
library(keras)

## Variables to easily access package functions
layers <- tf$keras$layers
np <- import("numpy")
bf <- import("bayesflow")
tfp <- import("tensorflow_probability")

## Load Auxiliary R functions
source("src/utils.R")

## Fix seed
set.seed(1)

## Set up parameters
ngrid <- 16L                        # Number of grid points in each dimension  
ngrid_squared <- as.integer(16^2)   # Number of grid points squared

## Set up spatial grid on [0, 1] x [0, 1]
s1 <- s2 <- seq(0, 1, length.out = ngrid)
sgrid <- expand.grid(s1 = s1, s2 = s2)

## Find distance matrix for these grid points
D <- fields::rdist(sgrid)

## Set up in TensorFlow and tile for automatic dispatch
D_tf <- tf$expand_dims(tf$constant(D, 
                        dtype = "float32"), 
                      0L)

## Define dist to be the normal distribution in tfp
dist = tfp$distributions$Normal(loc = 0, scale = 1)

#########################################
########## BayesFlow ####################
#########################################

## BayesFlow settings
bayesflow_iterations <- 3000L       # Number of BF iterations per epoch
bayesflow_epochs <- 2L              # Number of BF epochs
bayesflow_batch_size <- 32L         # Batch size for BF


## Fix BayesFlow RNG
RNG = np$random$default_rng(2023L)

## Set up the summary network
summary_net <- CNN(nconvs = 2L, 
                  ngrid = ngrid, 
                  kernel_sizes = c(3L, 3L),
                  filter_num = c(64L, 128L),
                  output_dims = 1L)

## For the inference network use an INN
inference_net <- bf$networks$InvertibleNetwork(
    num_params = 1L,
    num_coupling_layers = 4L,
    coupling_settings = list(dense_args = list(kernel_regularizer = NULL), 
                             dropout = FALSE)
)

## Load data
train_images <- readRDS("data/train_images.rds") %>% drop()
train_lscales <- readRDS("data/train_lscales.rds") %>% drop() %>% as.matrix()
val_images <- readRDS("data/val_images.rds") %>% drop()
val_lscales <- readRDS("data/val_lscales.rds") %>% drop() %>% as.matrix()

## Set up the amortizer in BayesFlow
simulation_dict <- list(sim_data = train_images,
                        prior_draws = trans_normCDF_inv(train_lscales))
validation_dict <- list(sim_data = val_images,
                         prior_draws = trans_normCDF_inv(val_lscales))

amortizer = bf$amortizers$AmortizedPosterior(inference_net, 
                                             summary_net)
trainer = bf$trainers$Trainer(amortizer = amortizer, 
                              checkpoint_path = "ckpts/BayesFlow/")

if(!is.null(trainer$manager$latest_checkpoint)) {
  cat("Loading pre-trained network...\n")
  history = trainer$load_pretrained_network()
} else {
  cat("Training the network...\n")
  history = trainer$train_offline(simulations_dict = simulation_dict,
                                epochs = bayesflow_epochs, 
                                batch_size = bayesflow_batch_size, 
                                validation_sims = validation_dict)
}

##############################################
########## Simulate test cases ###############
##############################################

cat("Applying to test data...\n")

## Run the amortizer on test data
test_images <- readRDS("data/test_images.rds")
test_micro_images <- readRDS("data/micro_test_images.rds")

## Posterior samples from BayesFlow on test cases
BayesFlow_synth_samples <- amortizer$sample(list(summary_conditions = test_images), 
                                   n_samples = 1000L) %>% 
                                   trans_normCDF()

## Posterior samples from BayesFlow on micro-test cases
BayesFlow_synth_micro_samples <- amortizer$sample(list(summary_conditions = test_micro_images), 
                                        n_samples = 1000L) %>% 
                                        trans_normCDF()

cat("Saving results...\n")
saveRDS(BayesFlow_synth_samples %>% drop(), "output/BayesFlow_test.rds")
saveRDS(BayesFlow_synth_micro_samples %>% drop(), "output/BayesFlow_micro_test.rds")
