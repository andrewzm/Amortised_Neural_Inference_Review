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

library(dplyr)
library(ggplot2)
library(tidyr)
library("gridExtra")
## Plots for naive and MI-based summary statistics
micro_test_lscales <- round(readRDS("data/micro_test_lscales.rds"), 5)
micro_test_images <- readRDS("data/micro_test_images.rds")

## Set up spatial grid on [0, 1] x [0, 1]
ngrid <- 16L                        # Number of grid points in each dimension  
s1 <- s2 <- seq(0, 1, length.out = ngrid)
sgrid <- expand.grid(s1 = s1, s2 = s2)

## Methods that sample from the posterior
preds <- list()
for(method in c("BayesFlow", "VB", "VB_Synthetic_Naive", 
                "VB_Synthetic_MutualInf", "Metropolis_Hastings")) {
   preds[[method]]  <- readRDS(paste0("output/", method, "_micro_test.rds"))
    
}

## NRE samples saved as numpy object, treat separately
library(reticulate)
np <- import("numpy")
preds[["NRE"]] <- np$load("output/NRE_micro_test.npy")

## Point summaries from Neural Bayes estimator
NBE <- read.csv("output/NBE_micro_test.csv")

zdf <- sgrid
samples_all <- NULL
for(i in 1:3) {
   zdf <- mutate(zdf, !!paste0("Z", i) := c(micro_test_images[i,,,]))
   samples_all <- rbind(samples_all,
                     sapply(preds, function(x) drop(x)[i,]) %>%
                     data.frame() %>%
                     mutate(lscale_true = micro_test_lscales[i]) %>%
                     gather(Method, l, -lscale_true))
   

}
zdf <- gather(zdf, sim, val, -s1, -s2)

## point summaries for other methods:
samples_all %>% group_by(lscale_true, Method) %>% summarise(Est = median(l))

spatplots <- ggplot(zdf) + geom_tile(aes(s1, s2, fill = val)) +
      scale_fill_distiller(palette = "Spectral") +
      facet_grid(~sim) +
      theme_bw() +
         theme(text = element_text(size = 7),
               legend.title = element_blank(),
               legend.position = "bottom") +
               coord_fixed()         

density_plots <- ggplot(samples_all) + 
           geom_density(aes(x = l, colour = Method), alpha = 0.5) + 
           geom_vline(data = NBE, aes(xintercept = estimate, colour = Method)) + 
           facet_wrap(~lscale_true, scales = "free_y") +
           geom_vline(aes(xintercept = lscale_true), 
                     linetype = "dashed", col = "black") +
           xlab("Length scale") + 
           ylab("Density") +
           xlim(c(0, 0.6)) +
           theme_bw() +
           theme(text = element_text(size = 10),
                 legend.title = element_blank(),
                 legend.position = "bottom")
  
g_all <- grid.arrange(grobs = list(spatplots, density_plots), ncol = 1)
ggsave("fig/micro_test_plots.png", g_all, width = 7, height = 7)
