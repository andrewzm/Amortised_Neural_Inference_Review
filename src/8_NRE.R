## Load packages and settings
source("src/setup.R")

## Get the settings for the current experiment
method <- "NRE"
settings <- config_setup(statmodel = statmodel, method = method)
p <- settings$num_params

# By default, NeuralEstimators will automatically find and utilise a working
# GPU, if one is available. To use the CPU (even if a GPU is available), set
# the following flag to FALSE.
use_gpu <- FALSE

# Start Julia with the project of the current directory:
Sys.setenv("JULIACONNECTOR_JULIAOPTS" = "--project=.")

juliaEval('
using NeuralEstimators
using BSON: @save, @load
using CSV
using DataFrames
using Flux
using RData
using Tables     
')

# ---- Training ----

estimator = juliaLet('
summary_network = Chain(
	Conv((5, 5), 1 => 6, relu),
	Conv((5, 5), 6 => 12, relu),
	Flux.flatten
	)
inference_network = Chain(
  Dense(768 + p, 50, relu), 
  Dense(50, 1)
  )
deepset = DeepSet(summary_network, inference_network)
estimator = RatioEstimator(deepset)
estimator
', p = p)

vectorise <- function(a) lapply(seq(dim(a)[4]), function(i) a[ , , , i, drop = F])
train_images <- readRDS(settings$fname_train_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
val_images   <- readRDS(settings$fname_val_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
train_params <- readRDS(settings$fname_train_params) %>% drop %>% as.matrix %>% t
val_params   <- readRDS(settings$fname_val_params) %>% drop %>% as.matrix %>% t

# train_images <- train_images[1:1000]
# val_images <- val_images[1:1000]
# train_params <- train_params[, 1:1000, drop = F]
# val_params <- val_params[, 1:1000, drop = F]

estimator <- juliaLet('
# By default, NeuralEstimators will automatically find and utilise a working
# GPU, if one is available. To use the CPU (even if a GPU is available), set
# the following flag to false.
use_gpu = false

train_images = broadcast.(Float32, train_images)
val_images   = broadcast.(Float32, val_images)
train_params = Float32.(train_params)
val_params   = Float32.(val_params)

@info "Training neural ratio estimator..."
estimator = train(estimator, train_params, val_params, train_images, val_images, use_gpu=use_gpu)

# Save the trained estimator
mkpath("ckpts/NRE")
estimator_state = Flux.state(estimator)
@save "ckpts/NRE/trained_estimators_$(statmodel).bson" estimator_state

# Load the saved estimator
#@load "ckpts/NRE/trained_estimators_$(statmodel).bson" estimator_state
#Flux.loadmodel!(estimator, estimator_state)

estimator
', estimator = estimator, statmodel = statmodel, train_images = train_images, train_params = train_params, val_images = val_images, val_params = val_params)


# ---- Testing ----

test_images <- readRDS(settings$fname_test_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
micro_test_images <- readRDS(settings$fname_micro_test_data) %>% aperm(c(2, 3, 4, 1)) %>% vectorise
test_images <- test_images[1:1000]

## Construct a parameter grid over which to evaluate the approx posterior density
steps <- 500
supports <- lapply(1:p, function(i) seq(settings$support_min[i], settings$support[i], length = steps)) 
theta_grid <- t(as.matrix(expand.grid(supports)))

## Number of Monte Carlo samples # TODO this should be a setting returned by config_setup
N <- 1000L 

MCsample <- function(estimator, images, N, theta_grid) {
  juliaLet('
   images = broadcast.(Float32, images)
   samples = sample.(Ref(estimator), images, N; theta_grid = theta_grid)
   samples = stackarrays(samples; merge = false)
   samples = permutedims(samples, (3, 2, 1))
  ', estimator = estimator, N = N, theta_grid = theta_grid, images = images)
}

micro_test_samples <- MCsample(estimator, micro_test_images, N, theta_grid)
test_samples <- MCsample(estimator, test_images, N, theta_grid)

## Sanity check: plotting
# i <- 1
# hist(micro_test_samples[i, , ])                                 # p = 1
# plot(micro_test_samples[i, , 1], micro_test_samples[i, , 2])    # p = 2 

cat("Saving results...\n")
saveRDS(micro_test_samples, settings$output_micro_path)
saveRDS(test_samples, settings$output_path)
