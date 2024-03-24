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
trans_normCDF <- function(x, support = 0.6) {
  if(length(support) > 1)
    if(length(dim(x)) == 2) {
        N <- dim(x)[1]
        support <- matrix(rep(support, N), nrow = N, byrow = TRUE)
    } else if(length(dim(x)) == 3) {
        N1 <- dim(x)[1]
        N2 <- dim(x)[2]
      support <- array(rep(support, N1*N2), dim = c(length(support), N2, N1)) %>%
                  aperm(c(3, 2, 1))
    }
  support * pnorm(x)
}

trans_normCDF_inv <- function(x, support = 0.6) {
    if(length(support) > 1)
        if(length(dim(x)) == 2) {
            N <- dim(x)[1]
            support <- matrix(rep(support, N), nrow = N, byrow = TRUE)
        } else if(length(dim(x)) == 3) {
            N1 <- dim(x)[1]
            N2 <- dim(x)[2]
            support <- array(rep(support, N1*N2), dim = c(length(support), N2, N1)) %>%
                      aperm(c(3, 2, 1))
        }
    qnorm(x/support)
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


# CNN <- function(nconvs, ngrid, kernel_sizes, filter_num,
#             lin_layer_size = 64L, method = "Point",
#             output_dims = 1L) {

#   ## Input layer
#   Z <- layer_input(shape = c(ngrid, ngrid, 1L))

#   ## Convolutional layers
#   encoders <- list()
#   for(i in 1:nconvs) {
#     ## First input is the data, subsequent inputs are the outputs of the previous layere
#     if(i == 1) input <- Z else input <- encoders[[i-1]]
#     ## Define the convolutional layers
#     encoders[[i]] <- layer_conv_2d(input,
#                             kernel_size = c(kernel_sizes[i], kernel_sizes[i]),
#                             filters = filter_num[i],
#                             strides = 1,
#                             activation = "relu",
#                             padding = "same",
#                             data_format = "channels_last") %>%
#     ## Define the max pooling layers
#     layer_max_pooling_2d(pool_size = c(2, 2), padding = "valid")

#   }

#   ## Flatten the output of the CNN
#   encoder_flatten <- layer_flatten(encoders[[nconvs]])

#   ## Add on dense layer
#   lin_layer <- layer_dense(encoder_flatten, lin_layer_size) %>%
#     layer_activation_leaky_relu()

#   ## Add on final layer that maps to the outputs
#   if(method == "Point") {
#     keras_model(Z, layer_dense(lin_layer, output_dims))
#   } else if(method == "VB") {
#     mean_VB <- layer_dense(lin_layer, output_dims)
#     log_sd_VB <- layer_dense(lin_layer, output_dims)
#     keras_model(Z, list(mean_VB, log_sd_VB))
#   } else {
#     stop("No appropriate method selected")
#   }
# }

# layer_sampler <- new_layer_class(
#   classname = "Sampler",
#   call = function(u_mean, u_log_sd) {
#     epsilon <- tf$random$normal(shape = tf$shape(u_mean))
#     u_mean + tf$exp(u_log_sd) * epsilon
#   }

# )



# model_vae <- new_model_class(
#   classname = "VAE",

#   initialize = function(encoder,
#                        synthetic = FALSE,
#                        summ_stat_compute = NA,
#                        summ_stat_network = list(),
#                        D_tf = NA, ...) {
#     super$initialize(...)
#     self$encoder <- encoder
#     self$sampler <- layer_sampler()
#     self$total_loss_tracker <-
#       metric_mean(name = "total_loss")
#     self$synthetic <- synthetic
#     self$summ_stat_compute = summ_stat_compute
#     self$summ_stat_network = summ_stat_network
#     self$summ_stat_network$trainable <- FALSE
#     self$D_tf <- D_tf
#   },

#   metrics = mark_active(function() {
#     list(
#       self$total_loss_tracker
#     )
#   }),

#   call = function(inputs, training = FALSE) {
#     c(u_trans_mean, u_trans_log_sd) %<-% self$encoder(inputs)
#     u <- self$sampler(u_trans_mean, u_trans_log_sd)
#     return(u)
#   },

#   train_step = function(data) {
#     with(tf$GradientTape() %as% tape, {
#         c(u_trans_mean, u_trans_log_sd) %<-% self$encoder(data)
#         u <- self$sampler(u_trans_mean, u_trans_log_sd)
#         l <- trans_sigmoid_tf(u)

#         var_VB <- tf$math$square(tf$exp(u_trans_log_sd))
#         log_q_density <- -u_trans_log_sd - 0.5*(u - u_trans_mean)^2 / var_VB

#         Z_long <- tf$reshape(data, c(-1L, ngrid_squared, 1L))

#         if(!self$synthetic) {
#             C <- tf$exp(- self$D_tf / tf$expand_dims(l, 2L))
#             L <- tf$linalg$cholesky(C)
#             logdiagL <- tf$math$log(tf$linalg$diag_part(L))
#             logdetpart <- -tf$reduce_sum(logdiagL, 1L)

#             LZ <- tf$linalg$triangular_solve(L, Z_long, lower = TRUE)
#             sqpart <- -0.5*tf$linalg$matmul(tf$linalg$matrix_transpose(LZ),
#                                             LZ)
#             total_loss <- tf$reduce_mean(log_q_density -logdetpart - sqpart)
#       } else {
#           summ_obs <- self$summ_stat_compute(data)
#           c(summ_mean, summ_log_sd) %<-% self$summ_stat_network(l)
#           summ_var <- tf$math$square(tf$exp(summ_log_sd))
#           log_p_density <- -summ_log_sd -0.5*(summ_obs - summ_mean)^2 / summ_var
#           total_loss <- tf$reduce_mean(
#                             tf$expand_dims(log_q_density,1L) - log_p_density)
#       }
#     })

#     grads <- tape$gradient(total_loss, self$trainable_weights)
#     self$optimizer$apply_gradients(zip_lists(grads, self$trainable_weights))

#     self$total_loss_tracker$update_state(total_loss)

#     list(total_loss = self$total_loss_tracker$result())
#   }
# )

CNN <- function(nconvs,
            ngrid,
            kernel_sizes,
            filter_num,
            lin_layer_size = 64L,
            method = "Point",
            n_comps = 1,
            dropout = FALSE,
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
                            padding = "same",
                            data_format = "channels_last") %>%
    ## Define the max pooling layers
    layer_max_pooling_2d(pool_size = c(2, 2), padding = "valid")

  }

  ## Flatten the output of the CNN
  encoder_flatten <- layer_flatten(encoders[[nconvs]])

  if(dropout == TRUE) {
    encoder_flatten <- tf$keras$layers$Dropout(0.3)(encoder_flatten)
  }


  ## Add on dense layer
  lin_layer <- layer_dense(encoder_flatten, lin_layer_size) %>%
    layer_activation_leaky_relu()

  ## Add on final layer that maps to the outputs
  if(method == "Point") {
    keras_model(Z, layer_dense(lin_layer, output_dims))
  } else if(method == "VB") {
      mean_VB <- layer_dense(lin_layer, output_dims * n_comps)
      log_sd_VB <- layer_dense(lin_layer, output_dims * n_comps)
      if(n_comps == 1) {
          keras_model(Z, list(mean_VB, log_sd_VB))
      } else {
        alpha_VB <- layer_dense(lin_layer, n_comps) %>% layer_activation_softmax()
        keras_model(Z, list(mean_VB, log_sd_VB, alpha_VB))
      }
  } else {
    stop("No appropriate method selected")
  }
}

layer_sampler <- new_layer_class(
  classname = "Sampler",
  call = function(params) {
    if(length(params) == 2) {  # sample from a single-component Gaussian

        u_mean <- params[[1]]
        u_log_sd <- params[[2]]
        epsilon <- tf$random$normal(shape = tf$shape(u_mean))
        u_mean + tf$exp(u_log_sd) * epsilon

    } else {  # sample from a mixture of Gaussians
        n_comps <- dim(params[[1]])[2]

        # if(n_comps > 2)
        #   stop("Currently only implement for n = 2 following the reparameterisation trick of THE CONCRETE DISTRIBUTION: A CONTINUOUS RELAXATION OF DISCRETE RANDOM VARIABLES by Maddison, Mnih and Teh")

        u_mean <- params[[1]]
        u_log_sd <- params[[2]]
        u_weights <- params[[3]]

        # alpha <- u_weights[, 2, drop = FALSE] / u_weights[, 1, drop = FALSE]
        # gamma <- tf$random$uniform(shape = c(tf$shape(u_weights)[1], 1L))
        # probs <- tf$math$log(alpha * gamma / (1 - gamma))
        # u_component1 <- (1 / (1 + tf$exp(-2 * 10 * probs)))
        # u_component2 <- 1 - u_component1
        # epsilon <- tf$random$normal(shape = tf$shape(u_mean))
        # u_component1 * (u_mean[,1, drop = FALSE] + tf$exp(u_log_sd[,1, drop = FALSE]) * epsilon[,1, drop = FALSE]) +
        # u_component2 * (u_mean[,2, drop = FALSE] + tf$exp(u_log_sd[,2, drop = FALSE]) * epsilon[,2, drop = FALSE])

        ## The following code works for n_comps > 2 but does not use the re-parameterisation trick and does not work
        u_component <- tf$random$categorical(tf$math$log(u_weights), 1L) %>%
                       tf$squeeze() %>%
                       tf$one_hot(depth = n_comps)
        epsilon <- tf$random$normal(shape = tf$shape(u_component))
        (u_component * (u_mean + tf$exp(u_log_sd) * epsilon)) %>% tf$reduce_sum(1L, keepdims = TRUE)


    }
  }

)


model_vae <- new_model_class(
  classname = "VAE",

  initialize = function(encoder,                        # encoder network
                       synthetic = FALSE,               # synthetic likelihood?
                       summ_stat_compute = NA,          # function for summary statistic
                       summ_stat_network = list(),      # network for summary statistic
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
    variational_params %<-% self$encoder(inputs)
    u <- self$sampler(variational_params)
    return(u)
  },

  train_step = function(data) {
    with(tf$GradientTape() %as% tape, {

        variational_params %<-% self$encoder(data)
        u <- self$sampler(variational_params)
        l <- trans_sigmoid_tf(u)

        u_trans_mean <- variational_params[[1]]
        u_trans_log_sd <- variational_params[[2]]
        sd_VB <- tf$exp(u_trans_log_sd)
        var_VB <- tf$math$square(sd_VB)

        if(length(variational_params) == 2) {
          log_q_density <- -u_trans_log_sd - 0.5*(u - u_trans_mean)^2 / var_VB
        } else {

          u_trans_weights <- variational_params[[3]]
          #  log_q_density <- X1 <- (u_trans_weights * ((1 / sd_VB) * tf$exp( -(u - u_trans_mean)^2 / (2*var_VB)))) %>%
          #                   tf$reduce_sum(1L, keepdims = TRUE) %>%
          #                   tf$math$log()
          exp_parts <- -tf$math$log(sd_VB) - (u - u_trans_mean)^2 / (2 * var_VB)
          max_exp_part <- tf$math$reduce_max(exp_parts, 1L, keepdims = TRUE)
          log_q_density <- (u_trans_weights * tf$exp(exp_parts - max_exp_part)) %>%
                           tf$reduce_sum(1L, keepdims = TRUE) %>%
                           tf$math$log() %>%
                           tf$math$add(max_exp_part)

        }




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

  call = function(inputs, training = FALSE) {
    return(self$S_net(inputs))
  },

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

config_setup <- function(case, method_name) {
  settings <- list()
  if(case == "GP") {
    settings$fname_train_data <- "data/train_images.rds"
    settings$fname_train_params <- "data/train_params.rds"
    settings$fname_val_data <- "data/val_images.rds"
    settings$fname_val_params <- "data/val_params.rds"
    settings$fname_test_data <- "data/test_images.rds"
    settings$fname_test_params <- "data/test_params.rds"
    settings$fname_micro_test_data <- "data/micro_test_images.rds"
    settings$fname_micro_test_params <- "data/micro_test_params.rds"
    settings$ckpt_path <- paste0("ckpts/", method_name, "/")
    settings$output_path <- paste0("output/", method_name, "_test.rds")
    settings$output_micro_path <- paste0("output/", method_name, "_micro_test.rds")
    settings$support <- 0.6
    settings$num_params <- 1L
  } else if(case == "MSP") {
    settings$fname_train_data <- "data/train_MSP_images.rds"
    settings$fname_train_params <- "data/train_MSP_params.rds"
    settings$fname_val_data <- "data/val_MSP_images.rds"
    settings$fname_val_params <- "data/val_MSP_params.rds"
    settings$fname_test_data <- "data/test_MSP_images.rds"
    settings$fname_test_params <- "data/test_MSP_params.rds"
    settings$fname_micro_test_data <- "data/micro_MSP_test_images.rds"
    settings$fname_micro_test_params <- "data/micro_MSP_test_params.rds"
    settings$ckpt_path <- paste0("ckpts/", method_name, "_MSP/")
    settings$output_path <- paste0("output/", method_name, "_MSP_test.rds")
    settings$output_micro_path <- paste0("output/", method_name, "_MSP_micro_test.rds")
    settings$support <- c(0.6, 3)
    settings$num_params <- 2L
  } else {
    stop("Method must be GP or MSP")
  }
  settings
}
