import torch
import numpy as np
import pickle # saving the trained estimator
import time
from rpy2.robjects import r, numpy2ri # used to load and save the data
from pathlib import Path
from sbi import utils as utils
from sbi.inference import simulate_for_sbi
from sbi.neural_nets.embedding_nets import CNNEmbedding

# This script illustrates the use of "amortised neural likeilihood-to-evidence
# ratio estimation", implemented in SBI as the method "SNRE_A". Many other
# amortised and sequential inference methods are available in the package, listed at:
# https://sbi-dev.github.io/sbi/tutorial/16_implemented_methods/
from sbi.inference import SNRE_A

# Which model are we using? Either the Gaussian process "GP", or inverted
# max-stable process "MSP"
model = "GP"
# for "historical" reasons, path for GP is the default
if model == "GP":
    model = ""
else:
    model = f"_{model}"

# Function to read RDS file and convert it to numpy array
def loaddata(file_path):
    rds_data = r['readRDS'](file_path) # load the RDS file
    np_array = np.array(rds_data)      # convert the R object to a numpy array
    torch_array = torch.from_numpy(np_array)
    torch_array = torch_array.float()
    return torch_array

train_images  = loaddata(f"data/train{model}_images.rds")
val_images    = loaddata(f"data/val{model}_images.rds")
test_images   = loaddata(f"data/test{model}_images.rds")
train_params = loaddata(f"data/train{model}_params.rds")
val_params   = loaddata(f"data/val{model}_params.rds")
test_params  = loaddata(f"data/test{model}_params.rds")
micro_test_params = loaddata(f"data/micro{model}_test_params.rds")
micro_test_images  = loaddata(f"data/micro{model}_test_images.rds")

# Construct the classifier...
# For a tutorial on CNN summary networks, see:
# https://github.com/sbi-dev/sbi/blob/main/tutorials/05_embedding_net.ipynb
embedding_net = CNNEmbedding(input_shape = (16, 16))
classifier = utils.classifier_nn(model="mlp", embedding_net_x = embedding_net, hidden_features=10)

# Prior
if "MSP" in model:
    prior_min = [0., 0.5]
    prior_max = [0.6, 3.0]
else:
    prior_min = [0.]
    prior_max = [0.6]

prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)

# Instantiate the inference object
inference = SNRE_A(prior, classifier = classifier)

# Add simulations to inference object
inference = inference.append_simulations(train_params, train_images)
inference = inference.append_simulations(val_params, val_images)

# Train the amortised likelihood-to-evidence ratio estimator
ratio_estimator = inference.train()

# Build the amortised posterior object
posterior = inference.build_posterior(ratio_estimator, prior = prior)

# Save the amortised posterior object
Path("ckpts/NRE").mkdir(parents=True, exist_ok=True)
file = open(f"ckpts/NRE/trained_estimator{model}.pkl", "wb")
pickle.dump(posterior, file)
file.close()

# Load the amortised posterior object
# file = open(f"ckpts/NRE/trained_estimator{model}.pkl", "rb")
# posterior = pickle.load(file)
# file.close()

#Function to evaluate the posterior density given a single image
def density_single_image(posterior, x, theta_grid):
    pdf = map(lambda theta: torch.exp(posterior.log_prob(theta,  x = x)), theta_grid)
    pdf = list(pdf)
    pdf = torch.cat(pdf)
    return pdf

# Function to evaluate the posterior density given a set of images
def density(posterior, images, theta_grid):
    pdf = map(lambda x: density_single_image(posterior, x, theta_grid = theta_grid), images)
    pdf = list(pdf)
    pdf = torch.stack(pdf)
    pdf = torch.Tensor.cpu(pdf)
    pdf = pdf.numpy()
    return pdf

# Evaluate over test sets
theta_grid = torch.linspace(0, 0.6, steps = 750)
micro_test_density = density(posterior, micro_test_images, theta_grid)
test_density = density(posterior, test_images[0:1000, :, :, :], theta_grid)

# Save output: .rds objects
# Enable automatic conversion of numpy arrays to R objects
numpy2ri.activate()
# Function to save numpy array as RDS file
def save_numpy_as_rds(np_array, file_path):
    # Convert numpy array to an R matrix
    if np_array.ndim == 1:
        r_matrix = r.matrix(np_array, nrow=np_array.shape[0], ncol=1)
    else:
        r_matrix = r.matrix(np_array, nrow=np_array.shape[0], ncol=np_array.shape[1])
    # Save the R matrix as an RDS file
    r['saveRDS'](r_matrix, file_path)
    return r_matrix

save_numpy_as_rds(theta_grid.numpy(), f"output/NRE{model}_theta_grid.rds")
save_numpy_as_rds(micro_test_density, f"output/NRE{model}_micro_test_density.rds")
save_numpy_as_rds(test_density, f"output/NRE{model}_test_density.rds")



# ---- unused code ----

# # Function to MCMC sample from the posterior given a set of images
# def sample(posterior, images, num_samples = 1000):
#     images  = np.split(images, images.shape[0]) # split 4D array into list of arrays
#     samples = map(lambda x: posterior.sample((num_samples,), x = x), images)
#     samples = list(samples)
#     samples = torch.cat(samples, 1)
#     samples = torch.permute(samples, (1, 0))
#     samples = torch.Tensor.cpu(samples)
#     samples = samples.numpy()
#     return samples

#t0 = time.time()
#pdf = density(posterior, micro_test_images, theta_grid)
#t1 = time.time()
#t = t1-t0 # 0.0003 seconds

#t0 = time.time()
#samples = sample(posterior, micro_test_images)
#t1 = time.time()
#t = t1-t0 # 24 seconds

# Find that evaluating the posterior density is much faster than MCMC sampling.
# Since we're considering a single-parameter model, we can estimate the density
# and use inverse-transform sampling, or similar, to generate samples from the
# posterior. This sampling will be done in a separate R script from convenience.

# Save output: .npy objects
# np.save(f"output/NRE{model}_micro_test.npy", micro_test_samples)
# np.save(f"output/NRE{model}_test.npy", test_density)
