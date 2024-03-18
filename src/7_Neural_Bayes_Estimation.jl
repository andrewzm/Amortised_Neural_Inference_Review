using NeuralEstimators
using BSON: @save, @load
using CSV
using DataFrames
using Distances
using Distributions
using Flux
using LinearAlgebra
using RData
using Tables

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
    data = [data[i, colons...] for i ∈ 1:size(data, 1)]
    return data
end

function loadparameters(filename)
	data = RData.load(joinpath("data", filename * ".rds"))
	data = Float32.(data) # convert to float32 for efficiency

	# The parameters are stored as Nx1xp arrays. However, NeuralEstimators
	# requires the parameters to be stored as pxN matrices. Since p=1 in this
	# example, this can be done with a straightforward reshape:
    data = reshape(data, 1, :)
    return data
end

train_images = loadimages("train_images")
val_images   = loadimages("val_images")
test_images  = loadimages("test_images")
micro_test_images  = loadimages("micro_test_images")

train_lscales = loadparameters("train_lscales")
val_lscales   = loadparameters("val_lscales")
test_lscales  = loadparameters("test_lscales")
micro_test_lscales = loadparameters("micro_test_lscales")

# ---- Construct the point and quantile estimators ----

p = 1 # number of parameters in the statistical model

dgrid = 16 # dimension of one side of grid

#channels = [64, 128]
channels = [32, 64]

# Summary network
ψ = Chain(
	Conv((3, 3), 1 => channels[1], relu, pad = SamePad()),
	MaxPool((2, 2)),
	Conv((3, 3),  channels[1] => channels[2], relu, pad = SamePad()),
	MaxPool((2, 2)),
	Flux.flatten
	)

# Inference network
ϕ = Chain(
  Dropout(0.1),
  Dense(dgrid * channels[end], 64, relu),
  Dropout(0.1),
  Dense(64, p)
  )

architecture = DeepSet(ψ, ϕ)

θ̂  = PointEstimator(architecture)
θ̂₂ = QuantileEstimator(architecture)

# ---- Train the estimators ----

@info "Training neural Bayes estimator: posterior mean"
θ̂ = train(θ̂, train_lscales, val_lscales, train_images, val_images, loss = Flux.mse)

@info "Training neural Bayes estimator: marginal posterior quantiles"
#θ̂₂ = train(θ̂₂, train_lscales, val_lscales, train_images, val_images)
N = 1000 # just for prototyping
θ̂₂ = train(θ̂₂, train_lscales[:, 1:N], val_lscales[:, 1:N], train_images[1:N], val_images[1:N])

# Assess the point estimator
assessment = assess(θ̂, val_lscales[:, 1:1000], val_images[1:1000])
bias(assessment)
rmse(assessment)
plot(assessment)

# Save the trained estimators
mkpath("ckpts/NBE")
point_estimator = Flux.state(θ̂)
quantile_estimator = Flux.state(θ̂₂)
@save "ckpts/NBE/trained_estimators.bson" point_estimator quantile_estimator

# Load the saved estimators
@load "ckpts/NBE/trained_estimators.bson" point_estimator quantile_estimator
Flux.loadmodel!(θ̂, point_estimator)
Flux.loadmodel!(θ̂₂, quantile_estimator)

# ---- Apply the estimators to the test data ----

priorsupport(θ̂) = min(max(θ̂, 0), 0.6)
priorsupport(θ̂::AbstractMatrix) = priorsupport.(θ̂)

function estimate(Z, θ̂, θ̂₂)

	# We can apply the estimators simply as θ̂(images), but here we use the
	# function estimateinbatches() to prevent memory issues when images is large
	point_estimates = estimateinbatches(Chain(θ̂, priorsupport), Z)
	quantile_estimates = estimateinbatches(Chain(θ̂₂, priorsupport), Z)

	# Store as N x 3 dataframe (point estimate, lower bound, upper bound)
	estimates = vcat(point_estimates, quantile_estimates)'
	estimates = DataFrame(estimates, ["estimate", "lower", "25thpercentile", "median", "75thpercentile", "upper"])
	estimates[:, :Method] .= "NBE" # save the method for plotting

	return estimates
end

estimates_micro_test = estimate(micro_test_images, θ̂, θ̂₂)
estimates_micro_test[:, :lscale_true] = vec(micro_test_lscales) # add true values
CSV.write("output/NBE_micro_test.csv", estimates_micro_test)

estimates_test = estimate(test_images, θ̂, θ̂₂)
estimates_test[:, :lscale_true] = vec(test_lscales) # add true values
CSV.write("output/NBE_test.csv", estimates_test)
