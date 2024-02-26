
library(fields)
library(dplyr)
library("reticulate")
use_condaenv("~/miniconda3/envs/BayesFlow")
library(tensorflow)
tfp <- import("tensorflow_probability")

set.seed(1)

## Total number of simulations (200,000)
n_batch <- 2000L
nsim_per_batch <- 100L
nsim_tot <- nsim_per_batch * n_batch
nsim_train<- round(0.8 * nsim_tot)
nsim_val <- round(0.1 * nsim_tot)
nsim_test <- nsim_tot - nsim_train - nsim_val

## Number of grid points
ngrid <- 16L  
ngrid_squared <- as.integer(16^2)

## Caltulate distances and set up for multiple dispatch
s1 <- s2 <- seq(0, 1, length.out = ngrid)
sgrid <- expand.grid(s1 = s1, s2 = s2)
D <- fields::rdist(sgrid)

D_tf <- tf$expand_dims(tf$constant(D, 
                        dtype = "float32"), 
                      0L)
D_sim_tf <- tf$tile(D_tf, c(nsim_per_batch, 1L, 1L))
    
## Initialise    
Z_sims_tf <- list()
lscales_tf <- list()
sum_stats_tf <- list()

## Data simulator
simulator <- function(lscales_tf, D_sim_tf = NA) {
    
    ## Number of simulations
    nsim <- dim(lscales_tf)[1]
    
    ## If D_sim_tf is not provided, compute it
    if (!is(D_sim_tf,  "tensorflow.tensor")) {
        D_sim_tf <- tf$tile(D_tf, c(nsim, 1L, 1L))
    }

    ## Construct covariance function
    C_sim_tf <- tf$exp(- D_sim_tf / lscales_tf)
    L_sim_tf <- tf$linalg$cholesky(C_sim_tf)

    ## Construct covariance function
    C_sim_tf <- tf$exp(- D_sim_tf / lscales_tf)
    L_sim_tf <- tf$linalg$cholesky(C_sim_tf)

    ## Simulate
    eta_sim <- array(rnorm(ngrid^2 * nsim),
                    dim = c(nsim, ngrid^2, 1L))
    eta_sim_tf <- tf$constant(eta_sim, 
                              dtype = "float32")
    Z_sims_long_tf <- tf$linalg$matmul(L_sim_tf, eta_sim_tf)
    
    ## Put data into array
    Z_sims_tf <- tf$reshape(Z_sims_long_tf,
                            c(nsim, ngrid, ngrid, 1L))
    return(Z_sims_tf)
}

## Simulate in batches for efficiency
for(i in 1:n_batch) {
    
    cat(paste0("Generating data in batch: ", i, "/", n_batch, "\n"))

    ## Simulate length scales
    lscales <- runif(n = nsim_per_batch, max = 0.6)
    lscales_tf[[i]] <- tf$constant(array(lscales, 
                            dim = c(nsim_per_batch, 1L, 1L)),
                            dtype = "float32")

    ## Put data into array
    Z_sims_tf[[i]] <- simulator(lscales_tf[[i]], D_sim_tf = D_sim_tf)

}

## Now collate the results and save
Z_sims_tf <- tf$concat(Z_sims_tf, 0L)
lscales_tf <- tf$concat(lscales_tf, 0L)

train_idx = 1:nsim_train
val_idx <- (nsim_train + 1):(nsim_train + nsim_val)
test_idx =  (nsim_train + nsim_val + 1):nsim_tot

train_images <- Z_sims_tf[train_idx,,,] %>% as.array()
val_images <- Z_sims_tf[val_idx,,,] %>% as.array()
test_images <- Z_sims_tf[test_idx,,,] %>% as.array()

train_lscales <- lscales_tf[train_idx,,] %>% as.array()
val_lscales <- lscales_tf[val_idx,,] %>% as.array()
test_lscales <- lscales_tf[test_idx,,] %>% as.array()

saveRDS(train_images, file = "data/train_images.rds")
saveRDS(val_images, file = "data/val_images.rds")
saveRDS(test_images, file = "data/test_images.rds")

saveRDS(train_lscales, file = "data/train_lscales.rds")
saveRDS(val_lscales, file = "data/val_lscales.rds")
saveRDS(test_lscales, file = "data/test_lscales.rds")


## Generate microtest data for the images
micro_test_lscales <- tf$constant(array(c(0.1, 0.3, 0.55), 
                            dim = c(3L, 1L, 1L)),
                            dtype = "float32")
micro_test_images <- simulator(micro_test_lscales)
saveRDS(micro_test_lscales %>% as.array(), file = "data/micro_test_lscales.rds")
saveRDS(micro_test_images %>% as.array(), file = "data/micro_test_images.rds")

