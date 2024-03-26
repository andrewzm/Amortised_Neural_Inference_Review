## NRE estimates of posterior density 
theta_grid <- readRDS("output/NRE_theta_grid.rds")
micro_test_density <- readRDS("output/NRE_micro_test_density.rds")
test_density <- readRDS("output/NRE_test_density.rds")

## Generate posterior samples
sample_NRE <- function(density, theta_grid) {
  sample(theta_grid, size = 1000, replace = TRUE, prob = density)
}
micro_test_samples <- t(apply(micro_test_density, 1, sample_NRE, theta_grid = theta_grid))
test_samples <- t(apply(test_density, 1, sample_NRE, theta_grid = theta_grid))

## Save samples
saveRDS(micro_test_samples, "output/NRE_micro_test.rds")
saveRDS(test_samples, "output/NRE_test.rds")
