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

## Core model construction
trans_normCDF <- function(x) {
    0.6 * pnorm(x)
}

trans_normCDF_inv <- function(x) {
    qnorm(x/0.6)
}

trans_normCDF_tf <- function(x) {
        tf$multiply(tf$constant(0.6, dtype = "float32"), 
                                 dist$cdf(x))
}

trans_sigmoid <- function(x) {
    0.6 / (1 + exp(-x))
}

trans_sigmoid_tf <- function(x) {
        tf$multiply(tf$constant(0.6, dtype = "float32"), 
                                 tf$math$sigmoid(x))
}


CNN <- function(nconvs, ngrid, kernel_sizes, filter_num, 
            lin_layer_size = 64L, method = "Point",
            output_dims = 1L) {

  ## Input layer
  Z <- layer_input(shape = c(ngrid, ngrid, 1L))

  ## Convolutional layers
  encoders <- list()
  for(i in 1:nconvs) {
    ## First input is the data, subsequent inputs are the outputs of the previous layere
    if(i == 1) input <- Z else input <- encoders[[i-1]]
    ## Define the convolutional layers
    encoders[[i]] <- layer_conv_2d(input,
                            kernel_size = c(kernel_sizes[i], kernel_sizes[i]), 
                            filters = filter_num[i],
                            strides = 1,
                            activation = "relu", 
                            padding = "valid",
                            data_format = "channels_last") %>%
    ## Define the max pooling layers
    layer_max_pooling_2d(pool_size = c(2, 2), padding = "valid")

  }
  
  ## Flatten the output of the CNN
  encoder_flatten <- layer_flatten(encoders[[nconvs]])

  ## Add on dense layer
  lin_layer <- layer_dense(encoder_flatten, lin_layer_size) %>%
    layer_activation_leaky_relu() 

  ## Add on final layer that maps to the outputs
  if(method == "Point") {
    keras_model(Z, layer_dense(lin_layer, output_dims))
  } else if(method == "VB") {
    mean_VB <- layer_dense(lin_layer, output_dims)
    log_sd_VB <- layer_dense(lin_layer, output_dims)
    keras_model(Z, list(mean_VB, log_sd_VB))
  } else {
    stop("No appropriate method selected")
  } 
}

layer_sampler <- new_layer_class(
  classname = "Sampler",
  call = function(u_mean, u_log_sd) {
    epsilon <- tf$random$normal(shape = tf$shape(u_mean))
    u_mean + tf$exp(u_log_sd) * epsilon
  }

)



model_vae <- new_model_class(
  classname = "VAE",

  initialize = function(encoder, 
                       synthetic = FALSE, 
                       summ_stat_compute = NA,
                       summ_stat_network = list(), 
                       D_tf = NA, ...) {
    super$initialize(...)
    self$encoder <- encoder
    self$sampler <- layer_sampler()
    self$total_loss_tracker <-
      metric_mean(name = "total_loss")
    self$synthetic <- synthetic
    self$summ_stat_compute = summ_stat_compute
    self$summ_stat_network = summ_stat_network
    self$summ_stat_network$trainable <- FALSE
    self$D_tf <- D_tf
  },
  
  metrics = mark_active(function() {
    list(
      self$total_loss_tracker
    )
  }),

  call = function(inputs, training = FALSE) {
    c(u_trans_mean, u_trans_log_sd) %<-% self$encoder(inputs)
    u <- self$sampler(u_trans_mean, u_trans_log_sd)
    return(u)
  },

  train_step = function(data) {
    with(tf$GradientTape() %as% tape, {
        c(u_trans_mean, u_trans_log_sd) %<-% self$encoder(data)
        u <- self$sampler(u_trans_mean, u_trans_log_sd)
        l <- trans_sigmoid_tf(u)
        
        var_VB <- tf$math$square(tf$exp(u_trans_log_sd))
        log_q_density <- -u_trans_log_sd - 0.5*(u - u_trans_mean)^2 / var_VB
        
        Z_long <- tf$reshape(data, c(-1L, ngrid_squared, 1L))
            
        if(!self$synthetic) {
            C <- tf$exp(- self$D_tf / tf$expand_dims(l, 2L))
            L <- tf$linalg$cholesky(C)
            logdiagL <- tf$math$log(tf$linalg$diag_part(L))
            logdetpart <- -tf$reduce_sum(logdiagL, 1L)
            
            LZ <- tf$linalg$triangular_solve(L, Z_long, lower = TRUE)
            sqpart <- -0.5*tf$linalg$matmul(tf$linalg$matrix_transpose(LZ),
                                            LZ)
            total_loss <- tf$reduce_mean(log_q_density -logdetpart - sqpart)
      } else {
          summ_obs <- self$summ_stat_compute(data) 
          c(summ_mean, summ_log_sd) %<-% self$summ_stat_network(l)
          summ_var <- tf$math$square(tf$exp(summ_log_sd))
          log_p_density <- -summ_log_sd -0.5*(summ_obs - summ_mean)^2 / summ_var
          total_loss <- tf$reduce_mean(
                            tf$expand_dims(log_q_density,1L) - log_p_density)
      }
    })

    grads <- tape$gradient(total_loss, self$trainable_weights)
    self$optimizer$apply_gradients(zip_lists(grads, self$trainable_weights))

    self$total_loss_tracker$update_state(total_loss)
   
    list(total_loss = self$total_loss_tracker$result())
  }
)


## Neural binding function network (mean and log sd)
model_summnet <- new_model_class(
    classname = "SummNet",

    initialize = function(summ_stat_network, ...) {
      super$initialize(...)
      self$summ_stat_network <- summ_stat_network
      self$total_loss_tracker <-
        metric_mean(name = "total_loss")
    },
    
    metrics = mark_active(function() {
      list(
        self$total_loss_tracker
      )
    }),

    train_step = function(data) {
      with(tf$GradientTape() %as% tape, {
            input_lscale <- data[[1]]
            output_summ <- data[[2]]
            c(summ_mean, summ_log_sd) %<-% self$summ_stat_network(input_lscale)
            summ_var <- tf$math$square(tf$exp(summ_log_sd))
            log_p_density <- -summ_log_sd -0.5*(output_summ - summ_mean)^2 / summ_var
            total_loss <- tf$reduce_mean(- log_p_density)
        })
      

      grads <- tape$gradient(total_loss, self$trainable_weights)
      self$optimizer$apply_gradients(zip_lists(grads, self$trainable_weights))

      self$total_loss_tracker$update_state(total_loss)
    
      list(total_loss = self$total_loss_tracker$result())
    }
  )

## Network for the summary statistic
model_MINet <- new_model_class(
  classname = "MINet",

  initialize = function(S_net, T_net, ...) {
    super$initialize(...)
    self$S_net <- S_net
    self$T_net <- T_net
    self$total_loss_tracker <-
      metric_mean(name = "total_loss")
  },
  
  metrics = mark_active(function() {
    list(
      self$total_loss_tracker
    )
  }),

  train_step = function(data) {
    with(tf$GradientTape() %as% tape, {
          Z <- data[[1]]
          l <- data[[2]][,1L,]
          lsim <- data[[2]][,2L,]
          S <- self$S_net(Z)
          l_S <- tf$concat(list(l, S), 1L) %>% tf$expand_dims(2L)
          lsim_S <- tf$concat(list(lsim, S), 1L) %>% tf$expand_dims(2L)
          T1 <- self$T_net(l_S)
          T2 <- self$T_net(lsim_S)
          total_loss <- tf$reduce_mean(
                 tf$math$softplus(T2) +
                 tf$math$softplus(-T1)
          )
      })
    

    grads <- tape$gradient(total_loss, self$trainable_weights)
    self$optimizer$apply_gradients(zip_lists(grads, self$trainable_weights))

    self$total_loss_tracker$update_state(total_loss)
   
    list(total_loss = self$total_loss_tracker$result())
  }
)