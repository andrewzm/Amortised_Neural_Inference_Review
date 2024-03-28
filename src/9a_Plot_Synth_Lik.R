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
library(grid)
library(gtable)

## Plots for naive and MI-based summary statistics

test_lscales <- readRDS("data/test_params.rds")

summ_data <- summ_test <- NULL
method_names <- list(Naive = "rKL2",
                     MutualInf = "rKL3")
for(method in c("Naive", "MutualInf")) {

    load(paste0("output/VB_Synthetic_", method, "_SummStat_data.rda"))
    df_for_plot$method <- method
    summ_data <- rbind(summ_data, df_for_plot)
    summ_test <- rbind(summ_test,
                        data.frame(l = test_lscales,
                                   s = test_summ_stat,
                                   method = method))

}

summ_data$method <- c(method_names[summ_data$method])
summ_data$method <- factor(summ_data$method,
                           levels = method_names)

summ_test$method <- c(method_names[summ_test$method])
summ_test$method <- factor(summ_test$method,
                           levels = method_names)

p <- ggplot(summ_data) +
    geom_point(data = summ_test, aes(l, s), col = "red", size = 0.2) +
    facet_wrap(~method, scales = "free", nrow = 2) +
    xlab(expression(theta)) +
    ylab(expression(S(bold(Z)))) +
    xlim(-0.1, 0.7) +
    theme_bw() +
    theme(text = element_text(size = 10),
            legend.title = element_blank()) +
    scale_x_continuous(expand = c(0.01, 0.01)) +
    scale_y_continuous(expand = c(0.01, 0.01)) +
    geom_line(aes(l, mu), col = "black") +
        geom_line(aes(l, mu + 1.95*sd), col = "blue", linetype = "dashed") +
        geom_line(aes(l, mu - 1.95*sd), col = "blue", linetype = "dashed") +
    labs(tag = "(a)") +
    theme(plot.tag = element_text(face = "bold", size = 10),
          plot.tag.position = c(0.02, 0.98))

ggsave(paste0("fig/synth_liks.pdf"), p, width = 2.6, height = 3.3)
