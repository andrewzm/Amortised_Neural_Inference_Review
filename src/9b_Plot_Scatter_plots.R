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

## Plots for naive and MI-based summary statistics
test_lscales <- readRDS("data/test_lscales.rds")
preds <- list()

## Methods that sample from the posterior
for(method in c("BayesFlow", "VB", "VB_Synthetic_Naive", 
                "VB_Synthetic_MutualInf")) {
   preds[[method]]  <- readRDS(paste0("output/", method, "_test.rds"))
}
point_summaries <- lapply(preds,
                          function(x) apply(x, 1, median)) %>%
                   data.frame() %>%
                   mutate(lscale_true = test_lscales) %>%
                   gather(Method, Est, -lscale_true)

## Add point summaries from Neural Bayes estimator
NBE <- read.csv("output/NBE_test.csv") %>% rename(Est = estimate) 
point_summaries <- bind_rows(point_summaries, NBE)

## Now make facet grid of scatter plots for each method
## with identity line
g <- ggplot(point_summaries) + 
    geom_point(aes(lscale_true, Est, col = Method), size = 0.2) +
    geom_abline(intercept = 0, slope = 1, col = "black") +
    xlab("True length scale") + ylab("Estimated length scale") +
    theme_bw() +
    coord_fixed() +
    scale_x_continuous(expand = c(0.01, 0.01)) + 
    scale_y_continuous(expand = c(0.01, 0.01)) +
    theme(text = element_text(size = 7),
          legend.title = element_blank()) +
    facet_wrap(~Method)

ggsave("fig/scatter_plots.png", g, width = 7, height = 7)
