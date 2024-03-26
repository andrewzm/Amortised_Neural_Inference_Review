using NeuralEstimators
using BSON: @save, @load
using CSV
using DataFrames
using Flux
using RData
using Tables

# Which model are we using? Either the Gaussian process "GP", or inverted
# max-stable process "MSP"
model = "MSP" # "MSP
path_modifier = model == "GP" ? "" : "_MSP"

# By default, NeuralEstimators will automatically find and utilise a working
# GPU, if one is available. To use the CPU (even if a GPU is available), set
# the following flag to false.
use_gpu = false

# ---- Load the data ----

function loadimages(filename)
	data = RData.load(joinpath("data", filename * ".rds"))
	data = Float32.(data) # convert to float32 for efficiency

	# In TensorFlow/Pytorch, images are stored as 4D arrays with dimension:
	# (N, width, height, channels) ≡ (N, 16, 16, 1)
    # However, in order to cater for independent replicates, NeuralEstimators
	# requires the images to be stored as N-vectors of 4D arrays, where each
	# array has dimension:
	# (width, height, channels, replicates) ≡ (16, 16, 1, 1).
    colons = ntuple(_ -> (:), ndims(data))
    [data[i, colons...] for i ∈ 1:size(data, 1)]
end

function loadparameters(filename)
	data = RData.load(joinpath("data", filename * ".rds"))
	data = Float32.(data) # convert to float32 for efficiency

	# The parameters are stored as Nxpx1 arrays. However, NeuralEstimators
	# requires the parameters to be stored as pxN matrices. This can be done by
	# dropping the third singleton dimension and permuting the dimensions:
	permutedims(dropdims(data; dims = 3))
end

train_images = loadimages("train$(path_modifier)_images")
val_images   = loadimages("val$(path_modifier)_images")
test_images  = loadimages("test$(path_modifier)_images")
micro_test_images  = loadimages("micro$(path_modifier)_test_images")

train_params = loadparameters("train$(path_modifier)_params")
val_params   = loadparameters("val$(path_modifier)_params")
test_params  = loadparameters("test$(path_modifier)_params")
micro_test_params = loadparameters("micro$(path_modifier)_test_params")

# ---- Construct the point and quantile estimators ----

p = size(train_params, 1) # number of parameters in the statistical model
dgrid = 16 # dimension of one side of grid
channels = [32, 64]

summary_network = Chain(
	Conv((3, 3), 1 => channels[1], relu, pad = SamePad()),
	MaxPool((2, 2)),
	Conv((3, 3),  channels[1] => channels[2], relu, pad = SamePad()),
	MaxPool((2, 2)),
	Flux.flatten
	)

inference_network = Chain(
  Dropout(0.1),
  Dense(dgrid * channels[end], 64, relu),
  Dropout(0.1),
  Dense(64, p)
  )

architecture = DeepSet(summary_network, inference_network)

θ̂  = PointEstimator(deepcopy(architecture))
θ̂₂ = PointEstimator(deepcopy(architecture))
θ̂₃ = IntervalEstimator(deepcopy(architecture); probs = [0.05, 0.95])

# ---- Train the estimators ----

@info "Training neural Bayes estimator: posterior mean"
θ̂ = train(θ̂, train_params, val_params, train_images, val_images, loss = Flux.mse, use_gpu=use_gpu)
@info "Training neural Bayes estimator: posterior median"
θ̂₂ = train(θ̂₂, train_params, val_params, train_images, val_images, use_gpu=use_gpu)
@info "Training neural Bayes estimator: marginal posterior quantiles"
θ̂₃ = train(θ̂₃, train_params, val_params, train_images, val_images, use_gpu=use_gpu)

# Assess the point estimator
assessment = assess(θ̂, val_params[:, 1:1000], val_images[1:1000], use_gpu=use_gpu)
bias(assessment)
rmse(assessment)
plot(assessment)

# Save the trained estimators
mkpath("ckpts/NBE")
mean_estimator = Flux.state(θ̂)
median_estimator = Flux.state(θ̂₂)
quantile_estimator = Flux.state(θ̂₃)
@save "ckpts/NBE/trained_estimators$(path_modifier).bson" mean_estimator median_estimator quantile_estimator

# Load the saved estimators
@load "ckpts/NBE/trained_estimators$(path_modifier).bson" mean_estimator median_estimator quantile_estimator
Flux.loadmodel!(θ̂, mean_estimator)
Flux.loadmodel!(θ̂₂, median_estimator)
Flux.loadmodel!(θ̂₃, quantile_estimator)

# ---- Apply the estimators to the test data ----

#TODO this will have to be updated for MSP
priorsupport(θ̂) = min(max(θ̂, 0), 0.6)
priorsupport(θ̂::AbstractMatrix) = priorsupport.(θ̂)

function estimate(Z, θ̂, θ̂₂)

	# We can apply the estimators simply as θ̂(images), but here we use the
	# function estimateinbatches() to prevent the possibility of memory issues
	# when images is large
	mean_estimates = estimateinbatches(Chain(θ̂, priorsupport), Z, use_gpu=use_gpu)
	median_estimates = estimateinbatches(Chain(θ̂₂, priorsupport), Z, use_gpu=use_gpu)
	quantile_estimates = estimateinbatches(Chain(θ̂₃, priorsupport), Z, use_gpu=use_gpu)

	# Store as dataframe
	estimates = vcat(mean_estimates, median_estimates, quantile_estimates)'
	estimates = DataFrame(estimates, ["estimate", "median", "lower", "upper"])
	estimates[:, :Method] .= "NBE" # save the method for plotting

	return estimates
end

estimates_micro_test = estimate(micro_test_images, θ̂, θ̂₂)
estimates_micro_test[:, :lscale_true] = vec(micro_test_params) # add true values
CSV.write("output/NBE_micro_test.csv", estimates_micro_test)

estimates_test = estimate(test_images, θ̂, θ̂₂)
estimates_test[:, :lscale_true] = vec(test_params) # add true values
CSV.write("output/NBE_test.csv", estimates_test)
