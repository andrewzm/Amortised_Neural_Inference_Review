library("ggplot2")
library("NeuralEstimators")
library("JuliaConnectoR")
library("dplyr")
library("latex2exp")
## Start Julia with the project of the current directory:
Sys.setenv("JULIACONNECTOR_JULIAOPTS" = "--project=.")

# Two possible models:
# homogeneous:    Z | θ ~ N(θ, s^2), θ ~ U(0, 1), s known 
# heterogeneous:  Z | θ ~ N(θ, θ^2), θ ~ U(0, 1)
heterogeneous <- TRUE

a <- 0
b <- 1
prior <- function(n) runif(n, min = a, max = b)
n <- 1000 # number of points used in the first plot
theta <- prior(n) 
thetapi <- theta[sample(1:n, n)]

simulate <- function(theta) {
  sd <- if(heterogeneous) theta else s
  rnorm(length(theta), mean = theta, sd = sd)
}
z <- simulate(theta)

df <- data.frame(
  theta = c(theta, thetapi), 
  class = rep(c("dependent", "independent"), each = n), 
  z = z
)
df <- df[sample(1:nrow(df)), ] # shuffle for plotting

g1 <- ggplot(df) + 
  geom_point(aes(x = theta, y = z, colour = class), alpha = 0.8, size = 0.7) + 
  labs(colour = "", x = expression(theta), y = "Z") + 
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0)) + 
  scale_colour_manual(values = c("blue", "red")) + 
  theme_bw() + 
  theme(legend.position = "top")

# Bayes optimal classifier: 
# c∗(Z, θ) = p(Z, θ){p(Z, θ) + p(Z)p(θ)}−1
#          = p(Z | θ)p(θ){p(Z | θ)p(θ) + p(Z)p(θ)}−1
#          = p(Z | θ){p(Z | θ) + p(Z)}−1
# Note that p(Z) = int p(Z | θ)p(θ) dθ, which can sometimes be evaluated in 
# closed form, and otherwise evaluated with Monte Carlo sampling.
Bayesclassifier_singlepair <- function(z, theta) {

  if (heterogeneous) {
    likelihood <- dnorm(z, mean = theta, sd = theta)
    theta_sample <- runif(10000, min = a, max = b)
    evidence <- mean(dnorm(z, mean = theta_sample, sd = theta_sample))
  } else {
    likelihood <- dnorm(z, mean = theta, sd = s)
    evidence <- pnorm(b, mean = z, sd = s) - pnorm(a, mean = z, sd = s)
  }
  
  likelihood / (likelihood + evidence)
}

df_grid <- expand.grid(theta = seq(a, b, length = 100), z = seq(min(z), max(z), length = 100))
df_grid$c <- apply(df_grid, 1, function(x) Bayesclassifier_singlepair(x["z"], x["theta"]))

g2 <- ggplot(df_grid) + 
  geom_raster(aes(x = theta, y = z, fill = c)) + 
  scale_fill_gradient2(low = 'red', mid = 'white', high = 'blue', limits = c(0, 1),
                       midpoint = 0.5, guide = 'colourbar', aesthetics = 'fill') +
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0)) + 
  labs(fill = "", x = expression(theta), y = "Z") + 
  theme_bw() + 
  theme(legend.position = "top")


# ---- Neural likelihood-to-evidence ratio ----

## Initialise the likelihood-to-evidence-ratio estimator, here using Julia code
## for added flexibility over the R helper function ?initialise_estimator
init_est <- function() {
  juliaEval('
  using NeuralEstimators, Flux, CUDA, cuDNN

  d = 1    # dimension of each replicate
  p = 1    # number of parameters in the statistical model
  w = 64   # number of neurons in each hidden layer

  summary_network = Chain(
  	Dense(d, w, relu),
  	Dropout(0.3),
  	Dense(w, w, relu),
  	Dropout(0.3),
  	Dense(w, w, relu),
  	Dropout(0.3)
  	)
  inference_network = Chain(
  	Dense(w + p, w, relu),
  	Dropout(0.3),
  	Dense(w, w, relu),
  	Dropout(0.3),
  	Dense(w, 1)
  	)
  deepset = DeepSet(summary_network, inference_network)
  RatioEstimator(deepset)
')
}
estimator <- init_est()

K <- 100000
theta_train <- prior(K) 
theta_val   <- prior(K)
Z_train     <- simulate(theta_train)
Z_val       <- simulate(theta_val)

## Coerce data and parameters to format required by NeuralEstimators
Z_train <- lapply(Z_train, as.matrix)
Z_val <- lapply(Z_val, as.matrix)
theta_train <- matrix(theta_train, nrow = 1)
theta_val <- matrix(theta_val, nrow = 1)

## Train the estimator
estimator <- train(estimator,
  theta_train = theta_train,
  theta_val   = theta_val,
  Z_train = Z_train,
  Z_val   = Z_val, 
  use_gpu = FALSE
)

## Apply the estimator to the test data
Z <- lapply(df_grid$z, as.matrix)
theta <- matrix(df_grid$theta, nrow = 1)
rhat <- estimate(estimator, Z, theta, use_gpu = FALSE)
chat <- rhat/(1+rhat)
df_grid$chat <- as.numeric(chat)

g3 <- ggplot(df_grid) + 
  geom_raster(aes(x = theta, y = z, fill = chat)) + 
  scale_fill_gradient2(low = 'red', mid = 'white', high = 'blue', limits = c(0, 1),
                       midpoint = 0.5, guide = 'colourbar', aesthetics = 'fill') +
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0)) + 
  labs(fill = "", x = expression(theta), y = "Z") + 
  theme_bw() + 
  theme(legend.position = "top")

figure <- egg::ggarrange(g1, g2, g3, nrow = 1)
ggsave("fig/Bayes_classifier.pdf", figure, width = 8.5, height = 3.5, device = "pdf")


# ---- Visualising the training procedure ----

cat("Visualising the neural Bayes classifier as a function of the training epoch... \n")

## initialise estimator as before
estimator <- init_est()

dfs <- list()
for (epoch in 0:9) {
  
  if (epoch > 0) {
    ## Train the estimator for one epoch
    estimator <- train(
      estimator, 
      theta_train = theta_train, theta_val = theta_val, 
      Z_train = Z_train, Z_val = Z_val, 
      epochs = 1, 
      use_gpu = FALSE
    )
  }
  
  ## Compute the current class probabilties
  rhat <- estimate(estimator, Z, theta, use_gpu = FALSE)
  chat <- rhat/(1+rhat)
  chat <- as.numeric(chat)

  ## Store information in data frame
  df <- df_grid
  df$chat <- chat
  df$epoch <- epoch
  dfs <- c(dfs, list(df))
}
df <- bind_rows(dfs)

g4 <- ggplot(df) + 
  geom_raster(aes(x = theta, y = z, fill = chat)) + 
  facet_wrap(epoch~., nrow = 2, labeller = label_bquote("epoch" == .(epoch))) + 
  scale_fill_gradient2(low = 'red', mid = 'white', high = 'blue', limits = c(0, 1),
                       midpoint = 0.5, guide = 'colourbar', aesthetics = 'fill') +
  scale_x_continuous(expand = c(0, 0), breaks = c(0.25, 0.5, 0.75)) + 
  scale_y_continuous(expand = c(0, 0)) + 
  labs(fill = "", x = expression(theta), y = "Z", title = TeX(r"(Neural classifier $c_{\bf{\gamma}}(\theta, Z)$)")) + 
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) 

g2 <- g2 + 
  labs(title = TeX(r"(Bayes classifier $c^*(\theta, Z)$)")) + 
  theme(legend.position = "none") + 
  theme(plot.title = element_text(hjust = 0.5))

figure <- egg::ggarrange(g2, g4, nrow = 1, widths = c(2, 5))

ggsave("fig/Bayes_classifier_vs_epoch.pdf", figure, width = 10.5, height = 3.5, device = "pdf")


## TODO 
## The problem is that our training set is so large that after a single epoch
## the classifier is essentially fully trained. Need to split the training set
## and show how the classifier changes within a single epoch.

n_batches <- 9
max_index <- 20000 # NB has to be less than K or there will be errors
indices <- split(1:max_index, cut(seq_along(1:max_index), n_batches, labels = FALSE))

## initialise estimator as before
estimator <- init_est()

dfs <- list()
for (epoch in 0:length(indices)) {
  
  if (epoch > 0) {
    
    idx <- indices[[epoch]]
    
    ## Train the estimator for one epoch
    estimator <- train(
      estimator, 
      theta_train = theta_train[, idx, drop=F], theta_val = theta_val[, idx, drop=F], 
      Z_train = Z_train[idx], Z_val = Z_val[idx], 
      epochs = 1, 
      verbose = FALSE,
      use_gpu = FALSE
    )
  }
  
  ## Compute the current class probabilties
  rhat <- estimate(estimator, Z, theta, use_gpu = FALSE)
  chat <- rhat/(1+rhat)
  chat <- as.numeric(chat)
  
  ## Store information in data frame
  df <- df_grid
  df$chat <- chat
  df$epoch <- epoch
  dfs <- c(dfs, list(df))
}
df <- bind_rows(dfs)
df$epoch <- df$epoch/n_batches * (max_index / K)
df$epoch <- round(df$epoch, 3)

g5 <- ggplot(df) + 
  geom_raster(aes(x = theta, y = z, fill = chat)) + 
  facet_wrap(epoch~., nrow = 2, labeller = label_bquote("epoch" == .(epoch))) + 
  scale_fill_gradient2(low = 'red', mid = 'white', high = 'blue', limits = c(0, 1),
                       midpoint = 0.5, guide = 'colourbar', aesthetics = 'fill') +
  scale_x_continuous(expand = c(0, 0), breaks = c(0.25, 0.5, 0.75)) + 
  scale_y_continuous(expand = c(0, 0)) + 
  labs(fill = "", x = expression(theta), y = "Z", title = TeX(r"(Neural classifier $c_{\bf{\gamma}}(\theta, Z)$)")) + 
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) 

figure <- egg::ggarrange(g2, g5, nrow = 1, widths = c(2, 5))

ggsave("fig/Bayes_classifier_vs_epochII.pdf", figure, width = 10.5, height = 3.5, device = "pdf")
                  