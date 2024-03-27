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
               method = "VB")

## Compile model with decoder (non-synthetic, true likelihood)
vae <- model_vae(phi_est, synthetic = FALSE, D_tf = D_tf)

## Load data
train_images <- readRDS("data/train_images.rds")

# Create a ModelCheckpoint callback
checkpoint_callback <- callback_model_checkpoint(
                              filepath = "./ckpts/NVI/checkpoint_epoch_{epoch}.hdf5",
                              save_weights_only = TRUE,
                              save_freq = 1, # Save after every epoch
                              verbose = 0)

## Train the model

if(file.exists("./ckpts/NVI/checkpoint_epoch_2.hdf5")) {
    vae %>% compile(optimizer = 'adam')
    dummy <- vae(train_images[1:2,,,]) # Just to initialise network
    vae %>% load_model_weights_hdf5("./ckpts/NVI/checkpoint_epoch_2.hdf5")
} else {
    vae %>% compile(optimizer = 'adam')
    history <- vae %>% fit(train_images, 
                        epochs = 2L,
                        shuffle = TRUE,
                        #batch_size = 128L, 
                        batch_size = 32L, 
                        callbacks = list(checkpoint_callback))
}


## Load and predict with test images
test_images <- readRDS("data/test_images.rds")
pred_VB_trans_mean <- vae$encoder$predict(test_images)[[1]]
pred_VB_trans_sd <- exp(vae$encoder$predict(test_images)[[2]])
VB_samples <- rnorm(n = 500 * dim(test_images)[1], 
                pred_VB_trans_mean, 
                pred_VB_trans_sd) %>%
            matrix(ncol = 500L) %>% trans_sigmoid()

## Load and predict with microtest images
test_micro_images <- readRDS("data/micro_test_images.rds")
pred_VB_trans_mean <- vae$encoder$predict(test_micro_images)[[1]]
pred_VB_trans_sd <- exp(vae$encoder$predict(test_micro_images)[[2]])
VB_samples_micro <- rnorm(n = 500 * dim(test_micro_images)[1], 
                pred_VB_trans_mean, 
                pred_VB_trans_sd) %>%
            matrix(ncol = 500L) %>% trans_sigmoid()

## Save results
saveRDS(VB_samples, "output/VB_test.rds")
saveRDS(VB_samples_micro, "output/VB_micro_test.rds")