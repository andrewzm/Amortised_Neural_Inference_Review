## Load packages and settings
source("src/setup.R")

## Get the settings for the current experiment
method <- "NBE"
settings <- config_setup(statmodel = statmodel, method = method)
p <- settings$num_params

# ---- Training ----

vectorise <- function(a) lapply(seq(dim(a)[4]), function(i) a[, , , i, drop = F])
train_images <- readRDS(settings$fname_train_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
val_images   <- readRDS(settings$fname_val_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
train_params <- readRDS(settings$fname_train_params) %>% drop %>% as.matrix %>% t
val_params   <- readRDS(settings$fname_val_params) %>% drop %>% as.matrix %>% t

## Use a subset of the training data for fast testing
# K <- 10000
# train_images <- train_images[1:K]
# val_images <- val_images[1:K]
# train_params <- train_params[, 1:K, drop = F]
# val_params <- val_params[, 1:K, drop = F]

architecture <- juliaLet("
  using NeuralEstimators, Flux, CUDA, cuDNN
	
  dgrid = 16 # dimension of one side of grid
  channels = [64, 128]
  summary_network = Chain(
  	Conv((3, 3), 1 => channels[1], relu),
  	MaxPool((2, 2)),
  	Conv((3, 3),  channels[1] => channels[2], relu),
  	MaxPool((2, 2)),
  	Flux.flatten, 
  	Dense(512, 10, relu), 
  	Dense(10, 10, relu), 
  	)
  	
  inference_network = Chain(
   	Dense(10, 10, relu),
    Dense(10, 10, relu),
    Dense(10, p)
    )

	DeepSet(summary_network, inference_network)
	", p = p)

estimator1 <- juliaLet("PointEstimator(architecture)", 
                       architecture = architecture)
estimator2 <- juliaLet("IntervalEstimator(architecture; probs = [0.05, 0.95])", 
                       architecture = architecture)

cat("Training neural Bayes estimator: posterior mean\n")
estimator1 <- train(estimator1, 
                    theta_train = train_params, theta_val = val_params, 
                    Z_train = train_images, Z_val = val_images, 
                    loss = "squared-error", 
                    savepath = paste0(settings$ckpt_path, "mean"), 
                    use_gpu = FALSE)

cat("Training neural Bayes estimator: posterior quantiles\n")
estimator2 <- train(estimator2, 
                    theta_train = train_params, theta_val = val_params, 
                    Z_train = train_images, Z_val = val_images,
                    savepath = paste0(settings$ckpt_path, "quantiles"), 
                    use_gpu = FALSE)

# ---- Testing ----

micro_test_images <- readRDS(settings$fname_micro_test_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
test_images <- readRDS(settings$fname_test_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
test_images <- test_images[1:1000]

compute_estimates <- function(estimator1, estimator2, images) {
  
  mean_estimates <- estimate(estimator1, images, use_gpu = FALSE)
  quantile_estimates <- estimate(estimator2, images, use_gpu = FALSE)
  estimates <- t(rbind(mean_estimates, quantile_estimates))
  a <- settings$support_min
  b <- settings$support
  estimates <- pmin(pmax(estimates, a), b)
  estimates <- data.frame(estimates)
  colnames(estimates) <- c("estimate", "lower", "upper")
  estimates$Method <- method 
  
  return(estimates)
}

estimates_micro_test <- compute_estimates(estimator1, estimator2, micro_test_images)
estimates_test <- compute_estimates(estimator1, estimator2, test_images)

# Add true values:
micro_test_params <- readRDS(settings$fname_micro_test_params) %>% drop %>% as.matrix %>% t
test_params <- readRDS(settings$fname_test_params) %>% drop %>% as.matrix %>% t
test_params <- test_params[, 1:1000, drop = FALSE]
estimates_micro_test$lscale_true <- c(micro_test_params)
estimates_test$lscale_true <- c(test_params)

cat("Saving results...\n")

saveRDS(estimates_micro_test, settings$output_micro_path)
saveRDS(estimates_test, settings$output_path)
