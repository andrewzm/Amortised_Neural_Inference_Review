# Estimation of length scale in Gaussian Process covariance function
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

## NRE estimates of posterior density 
theta_grid <- readRDS("output/NRE_theta_grid.rds")
micro_test_density <- readRDS("output/NRE_micro_test_density.rds")
test_density <- readRDS("output/NRE_test_density.rds")

## Generate posterior samples
sample_NRE <- function(density, theta_grid) {
  sample(theta_grid, size = 500, replace = TRUE, prob = density)
}
micro_test_samples <- t(apply(micro_test_density, 1, sample_NRE, theta_grid = theta_grid))
test_samples <- t(apply(test_density, 1, sample_NRE, theta_grid = theta_grid))

## Save samples
saveRDS(micro_test_samples, "output/NRE_micro_test.rds")
saveRDS(test_samples, "output/NRE_test.rds")
