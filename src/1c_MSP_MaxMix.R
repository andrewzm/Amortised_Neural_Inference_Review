library(SpatialExtremes)
library(fields)
library(geoR)
source("src/utils_MaxMix.R")

set.seed(1)

s <- as.matrix(expand.grid(seq(0, 1, length = 16),
                           seq(0, 1, length=16))) ## spatial coordinates on a 32x32 grid

test_images <- readRDS("data/micro_MSP_test_images.rds")
test_params <- readRDS("data/micro_MSP_test_params.rds")

pars <- list()
for(i in 1:3) {
    X <- t(as.vector(test_images[i,,,1] ))
    IMSP <- -1/log(1-exp(-X))
    image.plot(x = seq(0,1,length=16),
            y = seq(0,1,length=16),
            z = (matrix(log(IMSP[1,]), nrow = 16, ncol = 16)),
            xlab = "x", ylab = "y", main = "Max-stable")

    fit <- fitIMSP(data = IMSP,
                coord = s,
                distmax = 0.2,
                init = c(0.5,1.5)) ## Fitting the inverted MSP

    options(scipen = 3)
    pars[[i]] <- fit$par
    print(paste0("Sim ", i, " -- Estimated:", round(fit$par, 3), " Actual: ", test_params[i,,]), collapse = TRUE)
}

theta_est <- Reduce("rbind", pars)
saveRDS(theta_est, file = "output/CompLik_MSP_micro_test.rds")
