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

##############################################
## Summary Statistic extraction using MI #####
##############################################

## Load the data
train_images <- readRDS("data/train_images.rds") %>% tf$constant(dtype = "float32")
train_lscales <- readRDS("data/train_lscales.rds") %>% tf$constant(dtype = "float32")
val_images <- readRDS("data/val_images.rds") %>% tf$constant(dtype = "float32")
val_lscales <- readRDS("data/val_lscales.rds") %>% tf$constant(dtype = "float32")
test_images <- readRDS("data/test_images.rds") %>% tf$constant(dtype = "float32")
test_lscales <- readRDS("data/test_lscales.rds") %>% tf$constant(dtype = "float32")
micro_test_images <- readRDS("data/micro_test_images.rds") %>% tf$constant(dtype = "float32")

## Set up the networks
Snet <-  CNN(nconvs = 2L, 
            ngrid = ngrid, 
            kernel_sizes = c(3L, 3L),
            filter_num = c(64L, 128L), 
            output_dims = 2L)

Tnet_input <- layer_input(shape = c(3L)) %>%
    layer_batch_normalization()
Tdense_layer1 <- layer_dense(Tnet_input, 16L) %>%
    layer_activation_leaky_relu() 
Tdense_layer2 <- layer_dense(Tdense_layer1, 16L) %>%
    layer_activation_leaky_relu() 
T_output <- layer_dense(Tdense_layer2, 1L)
Tnet <- keras_model(Tnet_input, T_output)


summ_stat_input <- layer_input(shape = c(1L,1L))
dense_layer1 <- layer_dense(summ_stat_input, 16L) %>%
    layer_activation_leaky_relu() 
dense_layer2 <- layer_dense(dense_layer1, 16L) %>%
    layer_activation_leaky_relu() 
mean_summ_stat_net <- layer_dense(dense_layer2, 1L)
log_sd_summ_stat_net <- layer_dense(dense_layer2, 1L)
summ_stat_net <- keras_model(summ_stat_input, 
                      list(mean_summ_stat_net, log_sd_summ_stat_net))


phi_est <- CNN(nconvs = 2L, 
               ngrid = ngrid, 
               kernel_sizes = c(3L, 3L),
               filter_num = c(64L, 128L),
               method = "VB")

# Create a ModelCheckpoint callback for MI network
MI_ckpt_dir <-  "./ckpts/NVI_Synth_MutualInf_2SS/"
checkpoint_callback_MI <- callback_model_checkpoint(
                              filepath = paste0(MI_ckpt_dir, "/checkpoint_MI_epoch_{epoch}.hdf5"),
                              save_weights_only = TRUE,
                              verbose = 0)
MI_checkpoint = paste0(MI_ckpt_dir, "checkpoint_MI_epoch_5.hdf5")

# Create a ModelCheckpoint callback for VAE
VAE_ckpt_dir <-  "./ckpts/NVI_Synth_MutualInf_2SS/"
checkpoint_callback_VAE <- callback_model_checkpoint(
                              filepath = paste0(VAE_ckpt_dir, "/checkpoint_VAE_epoch_{epoch}.hdf5"),
                              save_weights_only = TRUE,
                              verbose = 0)
VAE_checkpoint <- paste0(VAE_ckpt_dir, "checkpoint_VAE_epoch_10.hdf5")

if(file.exists(VAE_checkpoint) & file.exists(MI_checkpoint)) {
   MIest <- model_MINet(S_net = Snet, T_net = Tnet)
   synth_lik_est <- model_summnet(summ_stat_net)
   vae_synth <- model_vae(phi_est, 
                    summ_stat_compute = function(Z)  MIest$S_net(Z) %>% tf$expand_dims(1L),
                    synthetic = TRUE,
                    summ_stat_network = synth_lik_est$summ_stat_network) 
   #MIest %>% compile(optimizer = 'adam')
   #synth_lik_est %>% compile(optimizer = 'adam')
   MIest %>% compile(optimizer = 'adam')
   vae_synth %>% compile(optimizer = optimizer_adam())
   dummy <- MIest(train_images[1:2,,,]) # Just to initialise network
   dummy <- vae_synth(train_images[1:2,,,]) # Just to initialise network
   MIest %>% load_model_weights_hdf5(MI_checkpoint)
   vae_synth %>% load_model_weights_hdf5(VAE_checkpoint)

} else {
    MIest <- model_MINet(S_net = Snet, T_net = Tnet)
    
    train_output_shuffled <- tf$random$shuffle(train_lscales, 0L)
    MIest %>% compile(optimizer = 'adam')
    MIest %>% fit(train_images, 
                tf$concat(list(train_lscales, 
                            train_output_shuffled), 
                            1L),
                epochs = 5L, 
                shuffle = TRUE,
                batch_size = 32L,
                callbacks = list(checkpoint_callback_MI))

    ## Obtain optimised summary statistics
    train_summ_stat <- MIest$S_net(train_images) %>% tf$expand_dims(2L)
    
    MIest$S_net$trainable <- FALSE

    ##############################################
    ###### Neural Binding Functions ##############
    ##############################################

    synth_lik_est <- model_summnet(summ_stat_net)
    
    synth_lik_est %>% compile(optimizer = 'adam')
    history <- synth_lik_est %>% fit(train_lscales, 
                                train_summ_stat, 
                                epochs = 10,
                                batch_size = 64L)
    synth_lik_est %>%  fit(train_lscales, 
                            train_summ_stat,
                            epochs = 10,
                            batch_size = 2048L)

    synth_lik_est$trainable <- FALSE
    synth_lik_est$summ_stat_network$trainable <- FALSE

    ## Save binding function data for plotting
    grid_l <- tf$constant(matrix(seq(0, 0.6, by = 0.001)))
    pred_mean <- synth_lik_est$summ_stat_network$predict(grid_l)[[1]]
    pred_sd <- (synth_lik_est$summ_stat_network$predict(grid_l)[[2]] %>% exp()) 
    df_for_plot <- data.frame(l = as.numeric(grid_l), 
                                mu = as.numeric(pred_mean), 
                                sd = as.numeric(pred_sd))

    
    ##############################################
    ####### NVI with Synthetic Likelihood ########
    ##############################################

    ## Train the VAE
    vae_synth <- model_vae(phi_est, 
                    summ_stat_compute = function(Z)  MIest$S_net(Z) %>% tf$expand_dims(1L),
                    synthetic = TRUE,
                    summ_stat_network = synth_lik_est$summ_stat_network)

    vae_synth %>% compile(
                optimizer = optimizer_adam(
                                learning_rate = 0.0002))
    
    history <- vae_synth %>% fit(train_images, 
                epochs = 10L, 
                shuffle = TRUE,
                batch_size = 512,
                callbacks = list(checkpoint_callback_VAE))
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

saveRDS(VB_samples, "output/VB_Synthetic_MutualInf_test.rds")
saveRDS(VB_samples, "output/VB_Synthetic_MutualInf_test.rds")
saveRDS(VB_samples_micro, "output/VB_Synthetic_MutualInf_micro_test.rds")

## Save summary statistics
train_summ_stat <- MIest$S_net(train_images) %>% tf$expand_dims(2L)    
val_summ_stat <- MIest$S_net(val_images) %>% tf$expand_dims(2L)
test_summ_stat <- MIest$S_net(test_images) %>% tf$expand_dims(2L)
micro_test_summ_stat <- MIest$S_net(micro_test_images) %>% tf$expand_dims(2L)

saveRDS(as.vector(train_summ_stat), "output/VB_Synthetic_MutualInf_train_SummStats.rds")
saveRDS(as.vector(val_summ_stat), "output/VB_Synthetic_MutualInf_val_SummStats.rds")
saveRDS(as.vector(test_summ_stat), "output/VB_Synthetic_MutualInf_test_SummStats.rds")
saveRDS(as.vector(micro_test_summ_stat), "output/VB_Synthetic_MutualInf_micro_test_SummStats.rds")
