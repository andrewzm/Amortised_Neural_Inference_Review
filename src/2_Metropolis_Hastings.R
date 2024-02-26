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

## Load data
test_images <- readRDS("data/test_images.rds")
test_micro_images <- readRDS("data/micro_test_images.rds")


#########################################
########## Metropolis-Hastings ##########
#########################################

## Setup
nMH <- 24000                        # Number of Metropolis-Hastings samples 
burnin <- 4000L                     # Burn-in period
thin <- 40L                         # Thinning factor

## Define the log of the multivariate normal density
logmvnorm <- function(l, Z) {

  ## x^T (LL^T)^-1 x
  ## = x^T L^{-T} L^{-1} x
  ## = y^T y where y = L^{-1}x = forwardsolve(L, x) 

  C <- exp(-D / l)
  L <- t(chol(C))
  -sum(log(diag(L))) - 0.5 * crossprod(forwardsolve(L, Z))
}

## For each test case
all_samples <- list()
for(j in 1:dim(test_micro_images)[1]) { 

  ## Extract true length scale and data
  Z <- test_micro_images[j,,,] %>%
       as.numeric()

  ## First sample
  lscale_samp <- rep(NA, nMH)
  lscale_samp[1] <- 0.6
  current_logmvnorm <- logmvnorm(lscale_samp[1], Z)
  naccept <- 0
  
  ## Run MCMC
  for(i in 2:nMH) {

    ## Propose 
    lscale_prop <- lscale_samp[i-1] + rnorm(n = 1, sd = 0.06)

    ## Accept/Reject
    if(lscale_prop < 0 | lscale_prop > 0.6) {
      alpha <- 0
    } else {
      new_logmvnorm <- logmvnorm(lscale_prop, Z)
      alpha <- exp(new_logmvnorm - current_logmvnorm)
    }
    u <- runif(1)
    if(u < alpha) {
      # Accept
      lscale_samp[i] <- lscale_prop
      current_logmvnorm <- new_logmvnorm
      naccept <- naccept + 1
    } else {
      ## Reject
      lscale_samp[i] <- lscale_samp[i-1]
    }

    ## Monitor acceptance rate
    if(i %% 1000 == 0)
      print(paste0("Sample ",i, ": Acceptance rate: ", naccept / i))

  }

  lscale_samp_thinned <- 
    lscale_samp[seq(1,nMH, by = thin)]
  all_samples[[j]] <- lscale_samp_thinned <- 
    lscale_samp_thinned[-(1:round(burnin / thin))]  
  print(paste0("Lag-1 correlation:", acf(lscale_samp_thinned)[[1]][2]))
}

saveRDS(do.call("rbind", all_samples), "output/Metropolis_Hastings_micro_test.rds")
