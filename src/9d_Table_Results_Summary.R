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
test_lscales <- readRDS("data/test_params.rds")[1:1000,,]
test_images <- readRDS("data/test_images.rds")[1:1000,,,]

summarise_method <- median

method_names <- list(Metropolis_Hastings= "MCMC",
                     NBE = "NBE",
                     BayesFlow = "fKL",
                     VB = "rKL1",
                     VB_Synthetic_Naive = "rKL2",
                     VB_Synthetic_MutualInf= "rKL3",
                     NRE = "NRE")

rmspe <- function(truth, est, summarise = mean) {
  sqrt(summarise((est - truth)^2))
}

mae <- function(truth, est, summarise = mean) {
   summarise(abs(est - truth))
}

is90 <- function(z, lower, upper, summarise = mean) {
   Score <- ((upper - lower) + 2/0.1 * (lower - z)*(z < lower) +
                    2/0.1 * (z - upper)*(z > upper))
   summarise(Score)

}

cov90 <- function(z, lower, upper, summarise = mean) {
    summarise((z < upper) & (z > lower))
}

crps <- function(z, samples, summarise = mean) {
   nsamples <- ncol(samples)
   col_shuffle <- sample(1:nsamples)
   z <- matrix(z, nrow = length(z), ncol = nsamples,
               byrow = FALSE)
   Score <- (rowMeans(abs(z - samples)) -
   0.5 * rowMeans(abs(z - samples[, col_shuffle])))
   summarise(Score)
}
preds <- results_crps <- NULL

for(method in c("Metropolis_Hastings",  "BayesFlow", "NRE", 
                "VB", "VB_Synthetic_MutualInf", "VB_Synthetic_Naive")) {
   preds[[method]]  <- drop(readRDS(paste0("output/", method, "_test.rds")))[1:1000, ]
   results_crps[[method]] <- crps(drop(test_lscales),
                                  drop(preds[[method]]),
                                  summarise = summarise_method)
}
results_crps <- data.frame(results_crps) %>% gather(Method, MCRPS)

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


NBE <- readRDS("output/NBE_test.rds") %>% rename(Est = estimate, Lower = lower, Upper = upper)
NBE <- NBE[1:1000, ]
all_results <- bind_rows(all_results, NBE)

all_results$Method <- c(method_names[all_results$Method])
all_results$Method <- factor(all_results$Method,
                                 levels = unlist(method_names))

mape_sd <- function(truth, est) {
  sd(abs(est - truth)) / sqrt(length(truth))
}

all_results <- mutate(all_results, Error = (lscale_true - Est))
all_results$Error_NMP <- filter(all_results, Method == "fKL")$Error
all_results$group <- cut(all_results$lscale_true,
                         breaks = seq(0, 0.6, by = 0.1))
g <- ggplot(filter(all_results, Method == "rKL1")) +
    geom_point(aes(x = Error, y = (abs(Error_NMP)) , colour = Method)) +
    geom_hline(aes(yintercept = 0)) + theme_bw()

g <- ggplot(all_results) +
    geom_smooth(aes(x = lscale_true, y = abs(Error), colour = Method), method = "loess") +
    geom_hline(aes(yintercept = 0)) + theme_bw()


densities_scores <- group_by(all_results, Method) %>%
      transmute(SPE = (lscale_true - Est)^2,
            IS90 = is90(lscale_true, Lower, Upper, summarise = I)) %>%
       mutate(IS90 = as.numeric(IS90)) %>%
       gather(Diagnostic, Value, -Method) %>%
       mutate(Diagnostic = factor(Diagnostic, levels = c("SPE", "IS90")))

g <- ggplot(densities_scores) +
    geom_density(aes(x = Value, colour = Method), linewidth = 0.3) +
    facet_wrap(~Diagnostic, scales = "free") +
    theme_bw() +
    scale_color_manual(values = c("MCMC" = "#1f77b4", 
                                "NBE"  = "#ff7f0e", 
                                "fKL"  = "#2ca02c", 
                                "rKL1" = "#d62728", 
                                "rKL2" = "#9467bd", 
                                "rKL3" = "#8c564b", 
                                "NRE"  = "#e377c2"),
                               breaks = levels(densities_scores$Method))
print(g)

ggsave("fig/scores_densities.pdf", g, width = 7.2, height = 3)


results_crps$Method <- c(method_names[results_crps$Method])
results_crps$Method <- factor(results_crps$Method,
                                 levels = sort(unlist(method_names)))


my_xtable <- group_by(all_results, Method) %>%
  summarise(RMSPE = rmspe(lscale_true, Est, summarise = summarise_method),
            MIS90 = is90(lscale_true, Lower, Upper, summarise = summarise_method),
            COV90 = cov90(lscale_true, Lower, Upper)) %>%
   left_join(results_crps, by = "Method") %>%
   select(1:3, 5, 4) %>%
   rotate_df(cn = 1) %>%
   xtable(digits = 2)
my_xtable[1:4, ] <- my_xtable[1:4,] *100
rownames(my_xtable)[1:3] <- paste(rownames(my_xtable)[1:3], "$(\\times 10^2)$")
rownames(my_xtable)[4] <- paste(rownames(my_xtable)[4], "$(\\%)$")
colnames(my_xtable) <- paste0("\\texttt{", colnames(my_xtable), "}")

latex <- my_xtable %>%  print(include.rownames = TRUE, only.contents = TRUE, 
                              sanitize.colnames.function = identity,
                              sanitize.rownames.function = identity)
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










