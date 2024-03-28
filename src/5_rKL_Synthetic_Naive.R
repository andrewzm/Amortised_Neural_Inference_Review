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
library(ggplot2)
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

## Find a mask for the distance matrix to be used in the summary statistic
D_mask_tf <- ((D_tf < (mean(diff(s1))*1.1)) %>%
              tf$where(1L, 0L) %>%
              tf$cast("float32"))

## "Naive" summary statistic: mean of squared differences between neighbours
summ_stat <- function(Z, D_mask, long = TRUE) {
  if(long == FALSE) {
    Z_long <- tf$reshape(Z, c(-1L, ngrid_squared, 1L))
  } else {
    Z_long <- Z
  }

  Z_long_t <- tf$linalg$matrix_transpose(Z_long)
  ((Z_long - Z_long_t) %>%
            tf$square() %>%
            tf$multiply(D_mask) %>%
            tf$reduce_mean(c(1L, 2L), keepdims = TRUE)) * 100
}

## VB Recognition model
phi_est <- CNN(nconvs = 2L,
               ngrid = ngrid,
               kernel_sizes = c(3L, 3L),
               filter_num = c(64L, 128L),
               method = "VB")

## Binding function model with naive summ statistic (fully connected)
summ_stat_input <- layer_input(shape = c(1L,1L))
  dense_layer1 <- layer_dense(summ_stat_input, 16L) %>%
      layer_activation_leaky_relu()
  dense_layer2 <- layer_dense(dense_layer1, 16L) %>%
      layer_activation_leaky_relu()
  mean_summ_stat_net <- layer_dense(dense_layer2, 1L)
  log_sd_summ_stat_net <- layer_dense(dense_layer2, 1L)
  summ_stat_net <- keras_model(summ_stat_input,
                        list(mean_summ_stat_net, log_sd_summ_stat_net))

## Train the binding function network
cat("Loading data\n")
train_images <- readRDS("data/train_images.rds")
train_lscales <- readRDS("data/train_params.rds")
val_images <- readRDS("data/val_images.rds")
val_lscales <- readRDS("data/val_params.rds")
test_images <- readRDS("data/test_images.rds")
test_lscales <- readRDS("data/test_params.rds")

if(file.exists("data/naive_summary_statistics.rds")) {
  cat("Loading naive summary statistics from file\n")
  train_summ_stat <- readRDS("data/naive_summary_statistics.rds") %>%
                     tf$constant(dtype = "float32")
} else {
   cat("Computing naive summary statistics\n")
  train_summ_stat <- summ_stat(train_images %>% tf$constant(dtype = "float32"),
                             D_mask_tf, long = FALSE)
  saveRDS(as.array(train_summ_stat), "data/naive_summary_statistics.rds")
}

cat("Fitting the binding function network\n")
synth_lik_est <- model_summnet(summ_stat_net)


# Create a ModelCheckpoint callback
checkpoint_callback <- callback_model_checkpoint(
                              filepath = "./ckpts/NVI_Synth_Naive/checkpoint_epoch_{epoch}.hdf5",
                              save_weights_only = TRUE,
                              save_freq = 1, # Save after every epoch
                              verbose = 0)

if(file.exists("./ckpts/NVI_Synth_Naive/checkpoint_epoch_10.hdf5")) {
   synth_lik_est %>% compile(optimizer = 'adam')
   vae_synth <- model_vae(phi_est,
                      summ_stat_compute = function(Z)  summ_stat(Z, D_mask_tf, long = FALSE),
                      synthetic = TRUE,
                      summ_stat_network = synth_lik_est$summ_stat_network)
    vae_synth %>% compile(optimizer = 'adam')
    dummy <- vae_synth(train_images[1:2,,,]) # Just to initialise network
    vae_synth %>% load_model_weights_hdf5("./ckpts/NVI_Synth_Naive/checkpoint_epoch_10.hdf5")
} else {

  synth_lik_est %>% compile(optimizer = 'adam')
  history <- synth_lik_est %>% fit(train_lscales,
                              train_summ_stat,
                              epochs = 10,
                              batch_size = 64L)
  synth_lik_est %>%  fit(train_lscales,
                          train_summ_stat,
                          epochs = 10,
                          batch_size = 2048L)

  ## Save binding function data for plotting
  test_summ_stat <- summ_stat(test_images %>% tf$constant(dtype = "float32"),
                              D_mask_tf, long = FALSE) %>% as.array()
  grid_l <- tf$constant(matrix(seq(0, 0.6, by = 0.001)))
  pred_mean <- synth_lik_est$summ_stat_network$predict(grid_l)[[1]]
  pred_sd <- (synth_lik_est$summ_stat_network$predict(grid_l)[[2]] %>% exp())
  df_for_plot <- data.frame(l = as.numeric(grid_l),
                              mu = as.numeric(pred_mean),
                              sd = as.numeric(pred_sd))
  save(test_summ_stat, df_for_plot,
          file = "output/VB_Synthetic_Naive_SummStat_data.rda")

  ## Incorporate within VAE
  cat("Incorporating binding function within VAE\n")
  vae_synth <- model_vae(phi_est,
                      summ_stat_compute = function(Z)  summ_stat(Z, D_mask_tf, long = FALSE),
                      synthetic = TRUE,
                      summ_stat_network = synth_lik_est$summ_stat_network)
  synth_lik_est$trainable <- FALSE
  synth_lik_est$summ_stat_network$trainable <- FALSE

    cat("Training the VAE\n")
    vae_synth %>% compile(
                  optimizer = optimizer_adam(
                                  learning_rate = 0.0002))

    history <- vae_synth %>% fit(train_images,
                  epochs = 10L,
                  shuffle = TRUE,
                  batch_size = 512L,
                  callbacks = list(checkpoint_callback))

}


## Testing with full test data
pred_VB_trans_mean <- vae_synth$encoder$predict(test_images)[[1]]
pred_VB_trans_sd <- exp(vae_synth$encoder$predict(test_images)[[2]])
VB_samples <- rnorm(n = 500 * dim(test_images)[1],
                pred_VB_trans_mean,
                pred_VB_trans_sd) %>%
            matrix(ncol = 500L) %>% trans_sigmoid()


## Load and predict with microtest images
test_micro_images <- readRDS("data/micro_test_images.rds")
pred_VB_trans_mean <- vae_synth$encoder$predict(test_micro_images)[[1]]
pred_VB_trans_sd <- exp(vae_synth$encoder$predict(test_micro_images)[[2]])
VB_samples_micro <- rnorm(n = 500 * dim(test_micro_images)[1],
                pred_VB_trans_mean,
                pred_VB_trans_sd) %>%
            matrix(ncol = 500L) %>% trans_sigmoid()

## Save results
saveRDS(VB_samples, "output/VB_Synthetic_Naive_test.rds")
saveRDS(VB_samples_micro, "output/VB_Synthetic_Naive_micro_test.rds")

