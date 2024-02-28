using BSON: @save, @load
using CSV
using DataFrames
using Distances
using Flux
using LinearAlgebra
using NeuralEstimators
using RData
using RecursiveArrayTools
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

# Summary network
ψ = Chain(
	Conv((3, 3), 1 => 32, relu),
	MaxPool((2, 2)),
	Conv((3, 3),  32 => 64, relu),
	MaxPool((2, 2)),
	Flux.flatten
	)

# Inference network
ϕ = Chain(Dense(256, 64, leakyrelu), Dense(64, p))

# DeepSet
architecture = DeepSet(ψ, ϕ)

# Initialise point estimator and posterior credible-interval estimator
g = Compress(0.0, 0.6) # optional function to ensure estimates fall within the prior support
θ̂ = PointEstimator(architecture, g)
θ̂₂ = IntervalEstimator(architecture, g; probs = [0.05, 0.95])
# TODO quantile estimator for 30 probability levels

# ---- Train the estimators ----

@info "Training the neural point estimator"
θ̂ = train(θ̂, train_lscales, val_lscales, train_images, val_images)

@info "Training the neural quantile estimator"
θ̂₂ = train(θ̂₂, train_lscales, val_lscales, train_images, val_images)

# Save the trained estimators
mkpath("ckpts/NBE")
point_estimator = Flux.state(θ̂)
quantile_estimator = Flux.state(θ̂₂)
@save "ckpts/NBE/trained_estimators.bson" point_estimator quantile_estimator

# Load the saved estimators
@load "ckpts/NBE/trained_estimators.bson" point_estimator quantile_estimator
Flux.loadmodel!(θ̂, point_estimator)
Flux.loadmodel!(θ̂₂, quantile_estimator)


# ---- Assess the estimators ----

assessment = assess(θ̂, val_lscales, val_images)
bias(assessment)
rmse(assessment)
plot(assessment)

assessment = assess(θ̂₂, val_lscales, val_images)
coverage(assessment)
# plot(assessment)


# ---- Apply the estimators to the test data ----

function estimate(Z, θ̂, θ̂₂)
	# We can apply the estimators simply as θ̂(images), but here we use the
	# function estimateinbatches() to prevent memory issues when images is large
	point_estimates = estimateinbatches(θ̂, Z)
	quantile_estimates = estimateinbatches(θ̂₂, Z)

	# Store as N x 3 dataframe (point estimate, lower bound, upper bound)
	estimates = vcat(point_estimates, quantile_estimates)'
	estimates = DataFrame(estimates, ["estimate", "lower", "upper"])
	estimates[:, :Method] .= "NBE" # save the method for plotting

	return estimates
end

estimates_micro_test = estimate(micro_test_images, θ̂, θ̂₂)
estimates_micro_test[:, :lscale_true] = vec(micro_test_lscales) # add true values
CSV.write("output/NBE_micro_test.csv", estimates_micro_test)

estimates_test = estimate(test_images, θ̂, θ̂₂)
estimates_test[:, :lscale_true] = vec(test_lscales) # add true values
CSV.write("output/NBE_test.csv", estimates_test)


## Bootstrap

function simulate(θ, m = 1)

	# Spatial locations
	pts = range(0, 1, length = 16)
	S = expandgrid(pts, pts)
	n = size(S, 1)

	# Distance matrix, covariance matrix, and Cholesky factor
	D = pairwise(Euclidean(), S, dims = 1)
	Σ = exp.(-D ./ θ)
	L = cholesky(Symmetric(Σ)).L

	# Spatial field
	Z = L * randn(n)

	# Reshape to 16x16 image and convert to Float32 for efficiency
	Z = reshape(Z, 16, 16, 1, 1)
	Z = Float32.(Z)

	return Z
end

function parametricbootstrap(θ̂, Z; B = 1000)
	point_estimate = θ̂(Z)
	Z_boot = [simulator(point_estimate) for _ ∈ 1:B]
	estimateinbatches(θ̂, Z_boot)
end

Z = micro_test_images
bs_estimates = parametricbootstrap.(Ref(θ̂), Z)
bs_estimates = vcat(bs_estimates...)
CSV.write("output/NBE_bootstrap_micro_test.csv", Tables.table(bs_estimates), writeheader=false)

Z = test_images
bs_estimates = parametricbootstrap.(Ref(θ̂), Z; B = 5)
bs_estimates = vcat(bs_estimates...)
CSV.write("output/NBE_bootstrap_test.csv", Tables.table(bs_estimates), writeheader=false)
