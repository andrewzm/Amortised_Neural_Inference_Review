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

library(fields)
library(dplyr)
library("reticulate")
use_condaenv("~/miniconda3/envs/BayesFlow")
library(tensorflow)
library(keras)
layers <- tf$keras$layers
np <- import("numpy")
bf <- import("bayesflow")
tfp <- import("tensorflow_probability")

## Load Auxiliary R functions
source("src/utils.R")
source("src/setup.R")

## Fix seed
set.seed(1)

## Set up parameters
dist = tfp$distributions$Normal(loc = 0, scale = 1)

## Set up parameters
ngrid <- 16L                        # Number of grid points in each dimension  
ngrid_squared <- as.integer(16^2)   # Number of grid points squared
n_comps <- 2L                       # Number of components in the mixture       

## Set up spatial grid on [0, 1] x [0, 1]
s1 <- s2 <- seq(0, 1, length.out = ngrid)
sgrid <- expand.grid(s1 = s1, s2 = s2)

## Find distance matrix for these grid points
D <- fields::rdist(sgrid)
D_tf <- tf$expand_dims(tf$constant(D, 
                        dtype = "float32"), 
                      0L)

## VB Recognition model
phi_est <- CNN(nconvs = 2L, 
               ngrid = ngrid, 
               kernel_sizes = c(3L, 3L),
               filter_num = c(64L, 128L),
               n_comps = n_comps,
               dropout = FALSE,
               method = "VB")

## Compile model with decoder (non-synthetic, true likelihood)
vae <- model_vae(phi_est, synthetic = FALSE, D_tf = D_tf)

## Load data
train_images <- readRDS("data/train_images.rds")

# Create a ModelCheckpoint callback
checkpoint_callback <- callback_model_checkpoint(
                              filepath = paste0("./ckpts/NVI_MDN/checkpoint_ncomps", n_comps ,"_epoch_{epoch}.hdf5"),
                              save_weights_only = TRUE,
                              save_freq = 1, # Save after every epoch
                              verbose = 0)

## Train the model

if(file.exists(paste0("./ckpts/NVI_MDN/checkpoint_ncomps", n_comps ,"_epoch_2.hdf5"))) {
    vae %>% compile(optimizer = 'adam')
    dummy <- vae(train_images[1:2,,,]) # Just to initialise network
    vae %>% load_model_weights_hdf5(paste0("./ckpts/NVI_MDN/checkpoint_ncomps", n_comps ,"_epoch_2.hdf5"))
} else {
    vae %>% compile(optimizer = "adam")
    history <- vae %>% fit(train_images, 
                        epochs = 2L,
                        shuffle = TRUE,
                        batch_size = 32L, 
                        callbacks = list(checkpoint_callback))
}

samples_MDN <- function(dataset, n_samples, params) {
    c(pred_mean, log_pred_sd, pred_weights) %<-% params
    pred_sd <- exp(log_pred_sd)
    n_comps <- dim(pred_mean)[2]
    samples <- matrix(NA, nrow = nrow(dataset), ncol = n_samples)
    for(i in 1:nrow(dataset)) {
        comp_select <- sample(1:n_comps, size = 500, pred_weights[i, ], replace = TRUE)     
        samples[i, ] <- rnorm(n = 500,
                              mean = pred_mean[i, comp_select],
                              sd =  pred_sd[i, comp_select])
    }
    samples
}


## Load and predict with test images
test_images <- readRDS("data/test_images.rds")
VB_params <- vae$encoder$predict(test_images)
VB_samples <- samples_MDN(test_images, 500, VB_params) %>% trans_sigmoid()


## Load and predict with microtest images
test_micro_images <- readRDS("data/micro_test_images.rds")
VB_params_micro <- vae$encoder$predict(test_micro_images)
VB_samples_micro <- samples_MDN(test_micro_images, 500, VB_params_micro) %>% trans_sigmoid()


# pred_VB_trans_mean <- VB_params[[1]]
# pred_VB_trans_sd <- exp(VB_params[[2]])
# pred_VB_trans_weights <- VB_params[[3]]
# VB_samples_micro <- matrix(NA, nrow = n_micro, ncol = 500)
# for(i in 1:n_micro) {
#         comp_select <- sample(1:n_comps, size = 500, pred_VB_trans_weights[i, ], replace = TRUE)     
#         VB_samples_micro[i, ] <- rnorm(n = 500,
#                               mean = pred_VB_trans_mean[i, comp_select],
#                               sd =  pred_VB_trans_sd[i, comp_select])  %>% trans_sigmoid()
# }

## Save results
saveRDS(VB_samples, "output/VB_MDN_test.rds")
saveRDS(VB_samples_micro, "output/VB_MDN_micro_test.rds")
