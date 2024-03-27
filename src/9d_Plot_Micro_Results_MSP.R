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
library("gridExtra")
## Plots for naive and MI-based summary statistics
micro_test_params <- readRDS("data/micro_MSP_test_params.rds")
micro_test_images <- readRDS("data/micro_MSP_test_images.rds")

## Set up spatial grid on [0, 1] x [0, 1]
ngrid <- 16L                        # Number of grid points in each dimension  
s1 <- s2 <- seq(0, 1, length.out = ngrid)
sgrid <- expand.grid(s1 = s1, s2 = s2)

## Methods that sample from the posterior
preds <- list()
method_names <- list(BayesFlow = "NF-NMP", NRE = "NRE")

for(method in c("BayesFlow", "NRE")) {
   preds[[method]]  <- readRDS(paste0("output/", method, "_MSP_micro_test.rds"))
}

preds %>% names

## Point summaries from Neural Bayes estimator
##NBE <- read.csv("output/NBE_micro_test.csv")

## CompLik
CompLikEst <- readRDS("output/CompLik_MSP_micro_test.rds") %>%
              as.data.frame() %>% rename(param1 = "V1", param2 = "V2")

zdf <- sgrid
samples_all <- NULL
for(i in 1:3) {
   zdf <- mutate(zdf, !!paste0("Z", i) := c(micro_test_images[i,,,]))
   for(j in seq_along(preds)) {
    x <- preds[[j]]
    dd <- data.frame(param1_true = micro_test_params[i, 1, 1],
                     param2_true = micro_test_params[i, 2, 1],
                     Method = names(preds)[j],
                     param1 = x[i,,1],
                     param2 = x[i,,2])
    samples_all <- rbind(samples_all, dd)
   }
  
}

zdf <- gather(zdf, sim, val, -s1, -s2)
zdf$simnum <- strsplit(zdf$sim, "Z") %>% sapply(function(x) as.numeric(x[2]))

samples_all$Method <- c(method_names[samples_all$Method])
samples_all$Method <- factor(samples_all$Method,
                                 levels = sort(unlist(method_names)))
samples_all$params_true <- paste0("(", round(samples_all$param1_true, 2), ", ", round(samples_all$param2_true, 2), ")")
CompLikEst$params_true <- unique(samples_all$params_true)                    
BayesFlowMean <- group_by(samples_all, params_true) %>% summarise(param1 = mean(param1), param2 = mean(param2)) %>% 
                 mutate(Method = "BayesFlow")

spatplots <- ggplot(zdf) + geom_tile(aes(s1, s2, fill = val)) +
      scale_fill_distiller(palette = "Spectral") +
      facet_wrap(~simnum, labeller = label_bquote(bold(Z)[.(simnum)])) +
      theme_bw() +
      theme(axis.text.x = element_blank(), # Remove x-axis labels
        axis.text.y = element_blank(), # Remove y-axis labels
        axis.ticks.x = element_blank(), # Remove x-axis ticks
        axis.ticks.y = element_blank(),  # Remove y-axis ticks 
        axis.title.x = element_blank(),  # Remove x-axis title
        axis.title.y = element_blank())  + # Remove y-axis title
      theme(text = element_text(size = 10),
            legend.title = element_blank(),
            legend.position = "left",
            legend.key.height= unit(0.2, 'in'),
            legend.key.width= unit(0.1, 'in'),
            legend.margin = margin(t = -18, r  = -10, unit = "pt"),
            panel.spacing = unit(1.2, "lines")) +
      coord_fixed() +  
      scale_x_continuous(expand = c(0, 0)) +
      scale_y_continuous(expand = c(0, 0)) +
      labs(tag = "(d)") +
      theme(plot.tag = element_text(face = "bold", size = 10),
          plot.tag.position = c(0.02, 1.1))


## Randomise plotting order 
samples_all <- samples_all[sample(1:nrow(samples_all)), ]

scatter_plots <- ggplot(samples_all) + 
           geom_point(aes(x = param1, y = param2, colour = Method), alpha = 1, size = 0.1) + 
           geom_point(aes(x = param1_true, y = param2_true), 
                        size = 1, col = "dark green") +
           geom_point(data = CompLikEst, aes(x = param1, y = param2), 
                      size = 1, col = "blue", shape = 3) +
           geom_point(data = BayesFlowMean, aes(x = param1, y = param2), 
                      size = 1, col = "dark red", shape = 3)  +
           facet_wrap(~ params_true, scales = "free_y",
                      labeller = label_bquote(theta[true] == .(params_true))) +
           xlab(expression(theta)) + 
           ylab("Density") +
           xlim(c(0, 0.6)) +
           theme_bw() +
           theme(text = element_text(size = 10),
                 legend.title = element_blank(),
                 legend.position = "bottom")

# Extract the legend
g <- ggplotGrob(scatter_plots)
legend <- gtable_filter(g, "guide-box")
# Save or display the legend separately
png("fig/micro_MSP_results_legend.png", width = 6, height = 0.5, 
   units = "in", res = 300) # Adjust size as needed
grid.draw(legend)
dev.off()

# Create a blank (white space) grob
blank_grob <- rectGrob(gp = gpar(col = NA, fill = "white"))

# Arrange the grobs with a small white space to the left of the top grob
g_all <- grid.arrange(
  #arrangeGrob(blank_grob, spatplots, ncol = 2, widths = c(1.5/20, 18.5/20)), # Adjust the ratio for the space
  spatplots, scatter_plots + theme(legend.position = "None"),
  nrow = 2, newpage = FALSE
) 

ggsave("fig/micro_MSP_test_plots.png", g_all, width = 3.6, height = 3.3)
