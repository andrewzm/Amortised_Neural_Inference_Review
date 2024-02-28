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

## Plots for naive and MI-based summary statistics

test_lscales <- readRDS("data/test_lscales.rds")

for(method in c("Naive", "MutualInf")) {

    load(paste0("output/VB_Synthetic_", method, "_SummStat_data.rda"))

    p <- ggplot(df_for_plot) +  
        geom_point(data = data.frame(l = as.numeric(test_lscales),
                                     s = as.numeric(test_summ_stat)),
                                    aes(l, s), col = "red", size = 0.2) +
        xlab("length scale") + ylab("summary statistic") +
                            xlim(-0.1, 0.7) +
        theme_bw() +
        theme(text = element_text(size = 10),
                legend.title = element_blank()) + 
        geom_line(aes(l, mu), col = "black") +
            geom_line(aes(l, mu + 1.95*sd), col = "blue", linetype = "dashed") + 
            geom_line(aes(l, mu - 1.95*sd), col = "blue", linetype = "dashed")
                        
        
    ggsave(paste0("fig/synth_lik_", method, ".png"), p, width = 8, height = 4)
}
