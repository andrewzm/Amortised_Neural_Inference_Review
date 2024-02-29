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
library(gridExtra)
library(xtable)

## Plots for naive and MI-based summary statistics
test_lscales <- readRDS("data/test_lscales.rds")[1:1000,,]
test_images <- readRDS("data/test_images.rds")[1:1000,,,]

rmspe <- function(truth, est) {
  sqrt(mean((est - truth)^2))
}

mae <- function(truth, est) {
   mean(abs(est - truth))
}

is90 <- function(z, lower, upper) {
   Score <- ((upper - lower) + 2/0.1 * (lower - z)*(z < lower) +
                    2/0.1 * (z - upper)*(z > upper)) %>%
            mean()

}

cov90 <- function(z, lower, upper) {
    mean((z < upper) & (z > lower))
}

crps <- function(z, samples) {
   nsamples <- ncol(samples)
   col_shuffle <- sample(1:nsamples)
   z <- matrix(z, nrow = length(z), ncol = nsamples,
               byrow = FALSE)
   (rowMeans(abs(z - samples)) - 
   0.5 * rowMeans(abs(z - samples[, col_shuffle]))) %>%
   mean()
}
preds <- results_crps <- NULL
for(method in c("Metropolis_Hastings", "BayesFlow", "VB", 
                "VB_Synthetic_MutualInf", "VB_Synthetic_Naive", "NRE")) {
   preds[[method]]  <- readRDS(paste0("output/", method, "_test.rds"))[1:1000, ]
   results_crps[[method]] <- crps(drop(test_lscales), 
                                  drop(preds[[method]]))
}
results_crps <- data.frame(results_crps) %>%
                gather(Method, CRPS)

all_results <- lapply(preds,
                          function(x) apply(x, 1, mean)) %>%
                   data.frame() %>%
                   mutate(lscale_true = test_lscales) %>%
                   gather(Method, Est, -lscale_true)

all_results$Lower <- lapply(preds,
                      function(x) apply(x, 1, quantile,0.05)) %>%
                   data.frame() %>%
                   gather(Method, Lower) %>%
                   pull(Lower)

all_results$Upper <- lapply(preds,
                      function(x) apply(x, 1, quantile,0.95)) %>%
                   data.frame() %>%
                   gather(Method, Upper) %>%
                   pull(Upper)

 all_results$Method <- factor(all_results$Method, 
                        levels = unique(all_results$Method))


latex <- group_by(all_results, Method) %>%
  summarise(rmspe = rmspe(lscale_true, Est),
             mae = mae(lscale_true, Est),
             is90 = is90(lscale_true, Lower, Upper),
             cov90 = cov90(lscale_true, Lower, Upper)) %>%
   left_join(results_crps, by = "Method") %>%
xtable(digits=4) %>% print()

writeLines(
  c(
    "\\documentclass[12pt]{article}",
    "\\begin{document}",
    "\\thispagestyle{empty}",
    latex,
    "\\end{document}"
  ),
  "fig/results_table.tex"
)

tools::texi2pdf("fig/results_table.tex", clean = TRUE)
file.rename(from = "results_table.pdf",  
            to = "fig/results_table.pdf")
