
library(fields)
library(dplyr)
library(SpatialExtremes)
library(parallel)

set.seed(1)

## Total number of simulations (200,000)
nsim_tot <- 181000L
nsim_train<- 160000L
nsim_val <- 20000L
nsim_test <- nsim_tot - nsim_train - nsim_val

## Number of grid points
ngrid <- 16L
ngrid_squared <- as.integer(16^2)

## Caltulate distances and set up for multiple dispatch
s1 <- s2 <- seq(0, 1, length.out = ngrid)
D <- fields::rdist(expand.grid(s1 = s1, s2 = s2))

## Initialise
Z_sims <- list()
lscales <- list()
sum_stats <- list()

## Data simulator
simulator <- function(params, s) {

    lscales <- params[, 1]
    smoothnesses <- params[, 2]

    Z <- mclapply(seq_along(lscales), function(i) {
                if(i %% 100 == 0) print(paste0("Arrived at ", i))
                1 / rmaxstab(1, coord = s, cov.mod = "whitmat",
                    nugget = 0, range = lscales[i],
                    smooth = smoothnesses[i], grid = TRUE)
                },
                mc.cores = 10L)

    return(Z)
}

## Generate microtest data for the images
micro_test_MSP_params <- matrix(c(0.05, 0.2, 0.4, 1.5,  1.4, 2.2), ncol = 2)
micro_test_MSP_params_arr <- array(micro_test_MSP_params, dim = c(3, 2, 1))
micro_test_MSP_images <- simulator(micro_test_MSP_params, s = cbind(s1, s2))
micro_test_MSP_images_arr <- simplify2array(micro_test_MSP_images)  %>% aperm(c(3,1,2)) %>%
             array(dim = c(3, ngrid, ngrid, 1L))
saveRDS(micro_test_MSP_images_arr, file = "data/micro_MSP_test_images.rds")
saveRDS(micro_test_MSP_params_arr, file = "data/micro_MSP_test_params.rds")

## Sample parameters
lscales <- runif(n = nsim_tot, min = 0, max = 0.6)
smoothnesses <- runif(n = nsim_tot, min = 0.5, max = 3)
params <- cbind(lscales, smoothnesses)
Z_sims <- simulator(params, s = cbind(s1, s2))

## Expand arrays by one dimension
params_arr <- array(params, dim = c(nsim_tot, 2, 1))
Z_sims_arr <- simplify2array(Z_sims)  %>% aperm(c(3,1,2)) %>%
             array(dim = c(nsim_tot, ngrid, ngrid, 1L))

train_idx = 1:nsim_train
val_idx <- (nsim_train + 1):(nsim_train + nsim_val)
test_idx =  (nsim_train + nsim_val + 1):nsim_tot

train_images <- Z_sims_arr[train_idx,,,,drop = FALSE]
val_images <- Z_sims_arr[val_idx,,,,drop = FALSE]
test_images <- Z_sims_arr[test_idx,,,,drop = FALSE]

train_params <- params_arr[train_idx,,,drop = FALSE]
val_params <- params_arr[val_idx,,,drop = FALSE]
test_params <- params_arr[test_idx,,,drop = FALSE]

saveRDS(train_images, file = "data/train_MSP_images.rds")
saveRDS(val_images, file = "data/val_MSP_images.rds")
saveRDS(test_images, file = "data/test_MSP_images.rds")

saveRDS(train_params, file = "data/train_MSP_params.rds")
saveRDS(val_params, file = "data/val_MSP_params.rds")
saveRDS(test_params, file = "data/test_MSP_params.rds")
