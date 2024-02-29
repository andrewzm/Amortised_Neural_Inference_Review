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

method_names <- list(Metropolis_Hastings= "MCMC", 
                     BayesFlow = "NF-NMP", 
                     VB = "TG-VB", 
                     VB_Synthetic_Naive = "TG-VB-Synth1", 
                     VB_Synthetic_MutualInf= "TG-VB-Synth2", 
                     NRE = "NRE",
                     NBE = "NBE")

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

for(method in c("Metropolis_Hastings",  "BayesFlow", "NRE",    
                "VB",  "VB_Synthetic_MutualInf", "VB_Synthetic_Naive")) {
   preds[[method]]  <- readRDS(paste0("output/", method, "_test.rds"))[1:1000, ]
   results_crps[[method]] <- crps(drop(test_lscales), 
                                  drop(preds[[method]]))
}
results_crps <- data.frame(results_crps) %>% gather(Method, CRPS)

all_results <- lapply(preds,
                          function(x) apply(x, 1, median)) %>%
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
 

NBE <- read.csv("output/NBE_test.csv") %>% rename(Est = estimate, Lower = lower, Upper = upper) 
NBE <- NBE[1:1000, ] 
all_results <- bind_rows(all_results, NBE)

all_results$Method <- c(method_names[all_results$Method])
all_results$Method <- factor(all_results$Method,
                                 levels = sort(unlist(method_names)))

results_crps$Method <- c(method_names[results_crps$Method])
results_crps$Method <- factor(results_crps$Method,
                                 levels = sort(unlist(method_names)))

my_xtable <- group_by(all_results, Method) %>%
  summarise(RMSPE = rmspe(lscale_true, Est),
            MAE = mae(lscale_true, Est),
            IS90 = is90(lscale_true, Lower, Upper),
            COV90 = cov90(lscale_true, Lower, Upper)) %>%
   left_join(results_crps, by = "Method") %>%
   arrange(RMSPE) %>%
xtable(digits = 3) 

latex <- my_xtable %>%  print(include.rownames = FALSE, only.contents = TRUE)
latex_standalone <- my_xtable %>%  print()
writeLines(
  c(
    "\\documentclass[12pt]{article}",
    "\\begin{document}",
    "\\thispagestyle{empty}",
    latex_standalone,
    "\\end{document}"
  ),
  "fig/results_table_standalone.tex"
)

writeLines(latex, "fig/results_table.tex")

tools::texi2pdf("fig/results_table_standalone.tex", clean = TRUE)
file.rename(from = "results_table_standalone.pdf",  
            to = "fig/results_table_standalone.pdf")
