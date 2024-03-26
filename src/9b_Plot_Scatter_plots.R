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
library(gtable)
library(grid)

## Plots for naive and MI-based summary statistics
test_lscales <- readRDS("data/test_lscales.rds")[1:1000,, ]
preds <- list()
method_names <- list(Metropolis_Hastings= "MCMC", 
                     BayesFlow = "fKL", 
                     VB = "rKL1", 
                     VB_Synthetic_Naive = "rKL2", 
                     VB_Synthetic_MutualInf= "rKL3", 
                     NRE = "NRE",
                     NBE = "NBE")

## Methods that sample from the posterior
for(method in c("Metropolis_Hastings", "BayesFlow", "VB", 
                "VB_Synthetic_Naive", 
                "VB_Synthetic_MutualInf", "NRE")) {
   preds[[method]]  <- readRDS(paste0("output/", method, "_test.rds"))
   preds[[method]] <- preds[[method]][1:1000, ] # Only keep 1000 test points
}

point_summaries <- lapply(preds,
                          function(x) apply(x, 1, mean)) %>%
                   data.frame() %>%
                   mutate(lscale_true = test_lscales) %>%
                   gather(Method, Est, -lscale_true)

## Add point summaries from Neural Bayes estimator
NBE <- read.csv("output/NBE_test.csv") %>% rename(Est = estimate) 
point_summaries <- bind_rows(point_summaries, NBE[1:1000, ])

point_summaries$Method <- c(method_names[point_summaries$Method])
point_summaries$Method <- factor(point_summaries$Method,
                                 levels = sort(unlist(method_names)))

## Now make facet grid of scatter plots for each method
## with identity line
p <- ggplot(point_summaries) + 
    geom_point(aes(lscale_true, Est, col = Method), size = 0.2) +
    geom_abline(intercept = 0, slope = 1, col = "black") +
    xlab(expression(theta)) + ylab(expression(hat(theta))) +
    theme_bw() +
    coord_fixed() +
    scale_x_continuous(expand = c(0.01, 0.01)) + 
    scale_y_continuous(expand = c(0.01, 0.01)) +
    theme(text = element_text(size = 10),
          legend.title = element_blank(),
          legend.position = "bottom",
          panel.spacing = unit(0.8, "lines")) +
      theme(strip.text = element_text(size = 6)) +
    facet_wrap(~Method, nrow = 1)  +
    labs(tag = "(c)") +
    theme(plot.tag = element_text(face = "bold", size = 10),
          plot.tag.position = c(0.02, 0.98))

# Extract the legend
g <- ggplotGrob(p)
legend <- gtable_filter(g, "guide-box")
# Save or display the legend separately
png("fig/scatter_plot_legend.png", width = 1000, height = 400, res = 300) # Adjust size as needed
grid.draw(legend)
dev.off()

ggsave("fig/scatter_plots.png", p + theme(legend.position = "none"), width = 7.2, height = 1.7)
