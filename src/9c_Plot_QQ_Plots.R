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

library(dplyr)
library(ggplot2)
library(tidyr)

## Plots for naive and MI-based summary statistics
test_lscales <- readRDS("data/test_lscales.rds")
preds <- list()

for(method in c("BayesFlow", "VB", "VB_Synthetic_Naive", 
                "VB_Synthetic_MutualInf")) {
   preds[[method]]  <- readRDS(paste0("output/", method, "_test.rds"))
    
}

quantile_grid <- seq(0, 1, by = 0.05)
quantiles <- lapply(preds,
                      function(method_results) {
                      sapply(quantile_grid, function(q) 
                         mean(test_lscales < apply(method_results, 1, quantile, q)))
                      })
quantiles_df <- data.frame(quantiles) %>%
                mutate(quantiles = quantile_grid) %>%
                gather(Method, Est, -quantiles)                   

## Now make quantile plots of all methods, with the identity line in red           
g <- ggplot(quantiles_df) + 
    geom_line(aes(quantiles, Est,  colour = Method)) +
    geom_abline(intercept = 0, slope = 1, col = "black") +
    xlab("Quantile") + ylab("Proportion of true length scale") +
    theme_bw() +
    theme(text = element_text(size = 10),
          legend.title = element_blank()) +
    coord_fixed(xlim = c(0,1), ylim = c(0,1))
              
ggsave("fig/quantile_plots.png", g, width = 6, height = 5)
