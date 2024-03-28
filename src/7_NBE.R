## Load packages and settings
source("src/setup.R")

## Get the settings for the current experiment
method <- "NBE"
settings <- config_setup(statmodel = statmodel, method = method)
p <- settings$num_params

# By default, NeuralEstimators will automatically find and utilise a working
# GPU, if one is available. To use the CPU (even if a GPU is available), set
# the following flag to FALSE.
use_gpu <- FALSE

juliaEval("
	using NeuralEstimators, Flux, CSV, DataFrames, Tables
	using BSON: @save, @load
	")

# ---- Training ----

vectorise <- function(a) lapply(seq(dim(a)[4]), function(i) a[, , , i, drop = F])
train_images <- readRDS(settings$fname_train_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
val_images   <- readRDS(settings$fname_val_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
train_params <- readRDS(settings$fname_train_params) %>% drop %>% as.matrix %>% t
val_params   <- readRDS(settings$fname_val_params) %>% drop %>% as.matrix %>% t

# train_images <- train_images[1:1000]
# val_images <- val_images[1:1000]
# train_params <- train_params[, 1:1000, drop = F]
# val_params <- val_params[, 1:1000, drop = F]

architecture <- juliaLet("
	summary_network = Chain(
	  Conv((5, 5), 1 => 6, relu),
	  Conv((5, 5), 6 => 12, relu),
	  Flux.flatten
	)
	inference_network = Chain(
	  Dense(768, 50, relu),
	  Dense(50, p)
	)
	DeepSet(summary_network, inference_network)
	", p = p)

estimators <- juliaLet('
	train_images = broadcast.(Float32, train_images)
	val_images   = broadcast.(Float32, val_images)
	train_params = Float32.(train_params)
	val_params   = Float32.(val_params)

	θ̂  = PointEstimator(architecture)
	θ̂₂ = IntervalEstimator(architecture; probs = [0.05, 0.95])

	@info "Training neural Bayes estimator: posterior mean"
	θ̂ = train(θ̂, train_params, val_params, train_images, val_images, loss = Flux.mse)

	@info "Training neural Bayes estimator: posterior quantiles"
	θ̂₂ = train(θ̂₂, train_params, val_params, train_images, val_images)

	θ̂, θ̂₂,
	',
    architecture = architecture, statmodel = statmodel,
    train_images = train_images, train_params = train_params,
    val_images = val_images, val_params = val_params,
    use_gpu = use_gpu
)

# ---- Testing ----

micro_test_images <- readRDS(settings$fname_micro_test_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
test_images <- readRDS(settings$fname_test_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
test_images <- test_images[1:1000]

compute_estimates <- function(estimators, images) {
    df <- juliaLet('
  θ̂, θ̂₂ = estimators
  ts = TruncateSupport(a, b)
  Z = broadcast.(Float32, Z)

  # We can apply the estimators simply as θ̂(images), but here we use the
  # function estimateinbatches() to prevent the possibility of memory issues
  mean_estimates = estimateinbatches(Chain(θ̂, ts), Z, use_gpu=use_gpu)
  quantile_estimates = estimateinbatches(Chain(θ̂₂, ts), Z, use_gpu=use_gpu)

  # Store as dataframe
  estimates = permutedims(vcat(mean_estimates, quantile_estimates))
  estimates = DataFrame(estimates, ["estimate", "lower", "upper"]) # TODO this needs to change when p > 1
  estimates[:, :Method] .= "NBE" # save the method for plotting
  estimates
	',
        estimators = estimators, 
        Z = images,
        a = settings$support_min, 
        b = settings$support,
        use_gpu = use_gpu
    )
    as.data.frame(df)
}

estimates_micro_test <- compute_estimates(estimators, micro_test_images)
estimates_test <- compute_estimates(estimators, test_images)

# Add true values:
micro_test_params <- readRDS(settings$fname_micro_test_params) %>% drop %>% as.matrix %>% t
test_params <- readRDS(settings$fname_test_params) %>% drop %>% as.matrix %>% t
test_params <- test_params[, 1:1000, drop = FALSE]
estimates_micro_test$lscale_true <- c(micro_test_params)
estimates_test$lscale_true <- c(test_params)

cat("Saving results...\n")

saveRDS(estimates_micro_test, settings$output_micro_path)
saveRDS(estimates_test, settings$output_path)
