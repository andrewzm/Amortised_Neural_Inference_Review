library(ggplot2)
library(dplyr)
library(gridExtra)

## Forward simulate through normalising flow (from base to target)
N <- 10000
u1 <- rnorm(N, 0, 1)
u2 <- rnorm(N, 0, 1)

alpha1 <- log(4)
mu1 <- 0
x1 <- u1 * exp(alpha1) + mu1

alpha2 <- 0
mu2 <- x1^2/4
x2 <- u2 * exp(alpha2) + mu2 
# 2d histogram with default option
data <- data.frame(x1 = x1, x2 = x2)
g1 <- ggplot(data, aes(x=x1, y=x2) ) +
  stat_density2d() +
  theme_bw() + xlim(c(-8, 8)) + ylim(c(-4, 10))

## Reverse simulate (from target to base)
u2_rev <- (x2 - mu2) / exp(-alpha2)
u1_rev <- (x1 - mu1) / exp(-alpha1)
g2 <- ggplot(data, aes(x=u1_rev, y=u2_rev) ) +
  stat_density2d() +
  theme_bw() 

## Analytic density given by change of variable formula
xgrid <- expand.grid(x1 = seq(-8, 8, by = 0.1), 
                     x2 = seq(-4, 10, by = 0.1)) %>% 
         mutate(u1 = (x1 - mu1) * exp(-alpha1),
                u2 = (x2 - x1^2/4) * exp(-alpha2),
                f =  exp(-(u1^2 + u2^2) / 2) * exp(-(alpha1 + alpha2)))

g3 <- ggplot(xgrid) + geom_tile(aes(x1, x2, fill = f)) + theme_bw()  + theme(legend.position = "none")
grid.arrange(g1, g2, g3, nrow = 2)

