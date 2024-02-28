import torch
import numpy as np
import pickle # saving the trained estimator
import time
from rpy2.robjects import r # used to load the data for the Gaussian-process example
from pathlib import Path
from sbi import utils as utils
from sbi.inference import simulate_for_sbi
from sbi.neural_nets.embedding_nets import CNNEmbedding

# This script illustrates the use of "amortised likeilihood-to-evidence ratio
# estimation", implemented in SBI as the method "SNRE_A". Many other amortised
# and sequential inference methods are available at:
# https://sbi-dev.github.io/sbi/tutorial/16_implemented_methods/
from sbi.inference import SNRE_A

# Function to read RDS file and convert it to numpy array
def loaddata(file_path):
    rds_data = r['readRDS'](file_path) # load the RDS file
    np_array = np.array(rds_data)      # convert the R object to a numpy array
    torch_array = torch.from_numpy(np_array)
    torch_array = torch_array.float()
    return torch_array

train_images  = loaddata("data/train_images.rds")
val_images    = loaddata("data/val_images.rds")
test_images   = loaddata("data/test_images.rds")
train_lscales = loaddata("data/train_lscales.rds")
val_lscales   = loaddata("data/val_lscales.rds")
test_lscales  = loaddata("data/test_lscales.rds")
micro_test_lscales = loaddata("data/micro_test_lscales.rds")
micro_test_images  = loaddata("data/micro_test_images.rds")

# Construct the classifier...
# For a tutorial on CNN summary networks, see:
# https://github.com/sbi-dev/sbi/blob/main/tutorials/05_embedding_net.ipynb
embedding_net = CNNEmbedding(input_shape = (16, 16))
classifier = utils.classifier_nn(model="mlp", embedding_net_x = embedding_net, hidden_features=10)

## found that the GPU did not speed things up, so stick with the cpu
# if torch.cuda.is_available():
#     device="cuda:0"
# else:
#     device="cpu:0"

# Prior
p = 1 # number of parameters
# prior = utils.BoxUniform(low=torch.zeros(p, device=device), high=0.6 * torch.ones(p, device=device))
prior = utils.BoxUniform(low=torch.zeros(p), high=0.6 * torch.ones(p))

# Instantiate the inference object
inference = SNRE_A(prior, classifier = classifier) #, device = device)

# Add simulations to inference object
inference = inference.append_simulations(train_lscales, train_images)
inference = inference.append_simulations(val_lscales, val_images)

# Train the amortised likelihood-to-evidence ratio estimator
ratio_estimator = inference.train()

# Build the amortised posterior object
posterior = inference.build_posterior(ratio_estimator, prior = prior)

# Save the amortised posterior object
Path("ckpts/NRE").mkdir(parents=True, exist_ok=True)
file = open("ckpts/NRE/trained_estimator.pkl", "wb")
pickle.dump(posterior, file)
file.close()

# Load the amortised posterior object
# file = open("ckpts/NRE/trained_estimator.pkl", "rb")
# posterior = pickle.load(file)
# file.close()


# Function to MCMC sample from the posterior given a set of images
def sample(posterior, images, num_samples = 500):
    images  = np.split(images, images.shape[0]) # split 4D array into list of arrays
    samples = map(lambda x: posterior.sample((num_samples,), x = x), images)
    samples = list(samples)
    samples = torch.cat(samples, 1)
    samples = torch.permute(samples, (1, 0))
    samples = torch.Tensor.cpu(samples)
    samples = samples.numpy()
    return samples

#Function to evaluate the posterior density given a single image
def density_single_image(posterior, x, steps):
    theta_list = torch.linspace(0, 0.6, steps = steps)
    pdf = map(lambda theta: torch.exp(posterior.log_prob(theta,  x = x)), theta_list)
    pdf = list(pdf)
    pdf = torch.cat(pdf)
    return pdf

# Function to evaluate the posterior density given a set of images
def density(posterior, images, steps = 250):
    pdf = map(lambda x: density_single_image(posterior, x, steps = steps), images)
    pdf = list(pdf)
    pdf = torch.stack(pdf)
    pdf = torch.Tensor.cpu(pdf)
    pdf.numpy()
    return pdf

t0 = time.time()
pdf = density(posterior, micro_test_images)
t1 = time.time()
t = t1-t0

t0 = time.time()
pdf = sample(posterior, micro_test_images)
t1 = time.time()
t = t1-t0

# Micro test set
# micro_test_density = density(posterior, micro_test_images)
# np.save("output/NRE_micro_test.npy", micro_test_density)
micro_test_samples = sample(posterior, micro_test_images, num_samples = 500)
np.save("output/NRE_micro_test.npy", micro_test_samples)

# Test set
test_density = density(posterior, test_images[1:1000, :, :, :])
np.save("output/NRE_test.npy", test_density)
