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

## Prior on unknown parameter is U(0,0.6)
prior_fun <- function() 
   RNG$uniform(low = 0.0, high = 0.6, size = 1L)    
prior = bf$simulation$Prior(prior_fun = prior_fun)

## Draw from this prior distribution to check it's working 
prior(batch_size = 10L)

## Define the data simulator for BayesFlow
likelihood_fun <- function(lscale) {

    ## Construct the covariance matrix and Cholesky factor for this parameter
    C_tf <- tf$exp(- D_tf / lscale)
    L_tf <- tf$linalg$cholesky(C_tf)

    ## Generate the N(0, 1) random variables
    eta_sim <- array(rnorm(ngrid^2),
                    dim = c(ngrid^2, 1L))

    ## Convert to TensorFlow
    eta_tf <- tf$constant(eta_sim, dtype = "float32")

    ## Generate the simulations by multiplying the Cholesky factor with the N(0, 1) random variables
    Z_sims_long_tf <- tf$linalg$matmul(L_tf, eta_tf)
    Z_sims_tf <- tf$reshape(Z_sims_long_tf,
                            c(ngrid, ngrid))
}

## Set up the simulator in BayesFlow
simulator <- bf$simulation$Simulator(simulator_fun = likelihood_fun)

## Set up the data simulation model in BayesFlow (prior + simulator)
model <- bf$simulation$GenerativeModel(prior = prior, simulator = simulator)

## Draw from the simulator to check it's working (check shapes)
out <- model(batch_size=3L)
print(paste0("Shape of sim_data: ", paste(dim(out$"sim_data"), collapse = " ")))

## Set up the summary network
summary_net <- CNN(nconvs = 2L, 
                  ngrid = ngrid, 
                  kernel_sizes = c(3L, 3L),
                  filter_num = c(64L, 128L),
                  output_dims = 1L)

## Test summary network (sizes, etc.)
test_inp = model(batch_size = 4L)
summary_rep = as.array(summary_net(test_inp$"sim_data"))
print("Shape of simulated data sets: ")
dim(test_inp$"sim_data")
print("Shape of summary vectors: ")
dim(summary_rep)

## For the inference network use an INN
inference_net <- bf$networks$InvertibleNetwork(
    num_params = 1L,
    num_coupling_layers = 4L,
    coupling_settings = list(dense_args = list(kernel_regularizer = NULL), 
                             dropout = FALSE)
)

## Test and see whether the inference network is functioning as it should
c(z, log_det_J) %<-% inference_net(test_inp$prior_draws, summary_rep)
print("Shape of latent variables:")
dim(as.array(z))
print("Shape of log det Jacobian:")
dim(as.array(log_det_J))

## Set up the amortizer in BayesFlow
amortizer = bf$amortizers$AmortizedPosterior(inference_net, summary_net)
trainer = bf$trainers$Trainer(amortizer = amortizer, generative_model = model)

## See what is contained in output of simulator
out = model(3L)
print("Keys of simulated dict: ")
names(out)

## See what is contained in output of trainer
conf_out <- trainer$configurator(out)
print("Keys of configured dict: ")
print(names(conf_out))

## Train the network
history = trainer$train_online(epochs = bayesflow_epochs, 
                               iterations_per_epoch = bayesflow_iterations, 
                               batch_size = bayesflow_batch_size, 
                               validation_sims = 200L)

## Plot history of training + validation
plot(history$"train_losses"[,1])
val_points <- seq(bayesflow_iterations, 
                  bayesflow_iterations * bayesflow_epochs, 
                  length.out = bayesflow_epochs)
lines(val_points, history$"val_losses"[,1], 
col = "red", 
lwd = 3)

##############################################
########## Simulate test cases ###############
##############################################

## Run the amortizer on test data
test_images <- readRDS("data/test_images.rds")
test_micro_images <- readRDS("data/micro_test_images.rds")

## Posterior samples from BayesFlow on test cases
BayesFlow_synth_samples <- amortizer$sample(list(summary_conditions = test_images), 
                                   n_samples = 500L)

## Posterior samples from BayesFlow on micro-test cases
BayesFlow_synth_micro_samples <- amortizer$sample(list(summary_conditions = test_micro_images), 
                                        n_samples = 500L)


saveRDS(BayesFlow_synth_samples, "output/BayesFlow_test.rds")
saveRDS(BayesFlow_synth_micro_samples, "output/BayesFlow_micro_test.rds")
