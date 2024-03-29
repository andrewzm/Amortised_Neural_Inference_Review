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
test_params <- readRDS("data/test_MSP_params.rds")[1:1000,, ]
preds <- list()
method_names <- list(BayesFlow = "fKL", 
                     CompLik = "MCL")

## Methods that sample from the posterior
for(method in c("BayesFlow")) {
   preds[[method]]  <- readRDS(paste0("output/", method, "_MSP_test.rds"))
   preds[[method]] <- preds[[method]][1:1000,, ] # Only keep 1000 test points
}

point_summaries <- lapply(seq_along(preds),
                          function(i) {
                             post_means <- apply(preds[[i]], c(1, 3), mean)
                             data.frame(Method = names(preds)[i],
                                        Param = rep(c("lambda", "nu"), each = 1000),
                                        Est = c(post_means[,1], post_means[,2]),
                                        Truth = c(test_params[,1],test_params[,2]))
                          }) %>%
                   Reduce("cbind", .)

## Add point summaries from point estimator
CompLik_raw <- readRDS("output/CompLik_MSP_test.rds") %>% as.vector()
CompLik <- data.frame(Est = CompLik_raw,
                      Truth = c(test_params[,1], test_params[,2]),
                      Param = rep(c("lambda", "nu"), each = 1000),
                      Method = "CompLik")

point_summaries <- bind_rows(point_summaries, CompLik)

point_summaries$Method <- c(method_names[point_summaries$Method])
point_summaries$Method <- factor(point_summaries$Method,
                                 levels = sort(unlist(method_names)))


equal_breaks <- function(n = 4, s = 0.5, ...){
  function(x){
    # rescaling
    d <- s * diff(range(x)) / (1+2*s)
    round(seq(min(x)+d, max(x)-d, length=n), 2)
  }
}

## Now make facet grid of scatter plots for each method
## with identity line
p <- ggplot(point_summaries) + 
    geom_point(aes(Truth, Est, col = Method), size = 0.2) +
    geom_abline(intercept = 0, slope = 1, col = "black") +
    xlab(expression(theta)) + ylab(expression(hat(theta))) +
    theme_bw() +
    #coord_fixed() +
    scale_x_continuous(expand = c(0.01, 0.01)) + 
    scale_y_continuous(expand = c(0.01, 0.01)) +
    theme(text = element_text(size = 10),
          legend.title = element_blank(),
          legend.position = "bottom",
          panel.spacing = unit(0.8, "lines")) +
      theme(strip.text = element_text(size = 6)) +
    facet_wrap(Method ~ Param, nrow = 2, scales = "free",
              labeller = label_parsed)  +
    labs(tag = "(b)") +
    theme(plot.tag = element_text(face = "bold", size = 10),
          plot.tag.position = c(0.02, 0.98)) +
    scale_x_continuous(breaks=equal_breaks(n=3, s=0.05), expand = c(0.05, 0))
    


# Extract the legend
g <- ggplotGrob(p)
legend <- gtable_filter(g, "guide-box")
# Save or display the legend separately
png("fig/scatter_plot_legend_MSP.png", width = 1000, height = 400, res = 300) # Adjust size as needed
grid.draw(legend)
dev.off()

ggsave("fig/scatter_plots_MSP.pdf", p + theme(legend.position = "none"), 
                width = 2.6, height = 3.3)

## Summaries
summaries <- group_by(point_summaries, Method, Param) %>%
             summarise(RMSE = sqrt(mean((Est - Truth)^2)))
                       
