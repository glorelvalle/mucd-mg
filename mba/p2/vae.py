# Implements auto-encoding variational Bayes.

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
    
from autograd import grad
from data import load_mnist
from data import save_images as s_images
from autograd.misc import flatten # This is used to flatten the params (transforms a list into a numpy array)

# images is an array with one row per image, file_name is the png file on which to save the images

# to stop code: import pdb; pdb.set_trace()

def save_images(images, file_name): return s_images(images, file_name, vmin = 0.0, vmax = 1.0)

# Sigmoid activiation function to estimate probabilities

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Relu activation function for non-linearity

def relu(x):    return np.maximum(0, x)

# This function intializes the parameters of a deep neural network

def init_net_params(layer_sizes, scale = 1e-2):

    """Build a (weights, biases) tuples for all layers."""

    return [(scale * npr.randn(m, n),   # weight matrix
             scale * npr.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

# This will be used to normalize the activations of the NN

# This computes the output of a deep neuralnetwork with params a list with pairs of weights and biases

def neural_net_predict(params, inputs):

    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""

    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)         # nonlinear transformation

    # Last layer is linear

    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb

    return outputs

# This implements the reparametrization trick

def sample_latent_variables_from_posterior(encoder_output):

    # Params of a diagonal Gaussian.

    D = np.shape(encoder_output)[-1] // 2
    mean, log_std = encoder_output[:, :D], encoder_output[:, D:]

    # Generate one sample from q(z|x) per each batch datapoint
    Z = mean + np.exp(log_std)*npr.randn(*mean.shape)

    return Z

# This evlauates the log of the term that depends on the data

def bernoulli_log_prob(targets, logits):

    # logits are in R
    # Targets must be between 0 and 1

    # Compute the log probability of the targets given the generator output specified in logits
    # Sum the probabilities across the dimensions of each image in the batch

    probs = sigmoid(logits)
    log_prob = np.sum(
                    np.log(
                        targets*probs + (1-targets)*(1-probs)
                    ),
                    axis = -1
                )

    return log_prob

# This evaluates the KL between q and the prior

def compute_KL(q_means_and_log_stds):
    
    D = np.shape(q_means_and_log_stds)[-1] // 2
    mean, log_std = q_means_and_log_stds[:, :D], q_means_and_log_stds[:, D:]

    # Compute the KL divergence between q(z|x) and the prior as a standard Gaussian
    KL_divergence = np.sum(
                        0.5*(np.exp(2*log_std) + mean**2 - 1 - 2*log_std), 
                        axis = -1
                    )

    return KL_divergence

# This evaluates the lower bound

def vae_lower_bound(gen_params, rec_params, data):
    # Compute a noisy estiamte of the lower bound by using a single Monte Carlo sample:

    # 1 - compute the encoder output using neural_net_predict given the data and rec_params
    encoder_output = neural_net_predict(rec_params, data)

    # 2 - sample the latent variables associated to the batch in data 
    latent_variables_samples = sample_latent_variables_from_posterior(encoder_output)

    # 3 -  Reconstruct the image using the sampled latent variables and compute the log_prob of the actual data
    decoder_output = neural_net_predict(gen_params, latent_variables_samples)
    log_prob = bernoulli_log_prob(data, decoder_output)

    # 4 - compute the KL divergence between q(z|x) and the prior
    KL_divergence = compute_KL(encoder_output)

    # 5 - return an average estimate (per batch point) of the lower bound by substracting the KL to the data dependent term
    estimated_lower_bound = np.mean(
                                log_prob - KL_divergence,
                                axis = -1
                            )

    return estimated_lower_bound


if __name__ == '__main__':

    # Model hyper-parameters

    npr.seed(0) # We fix the random seed for reproducibility

    latent_dim = 50
    data_dim = 784  # How many pixels in each image (28x28).
    n_units = 200
    n_layers = 2

    gen_layer_sizes = [ latent_dim ] + [ n_units for i in range(n_layers) ] + [ data_dim ]
    rec_layer_sizes = [ data_dim ]  + [ n_units for i in range(n_layers) ] + [ latent_dim * 2 ]

    # Training parameters

    batch_size = 200
    num_epochs = 30
    learning_rate = 0.001

    print("Loading training data...")

    N, train_images, _, test_images, _ = load_mnist()

    # Parameters for the generator network p(x|z)

    init_gen_params = init_net_params(gen_layer_sizes)

    # Parameters for the recognition network p(z|x)

    init_rec_params = init_net_params(rec_layer_sizes)

    combined_params_init = (init_gen_params, init_rec_params) 

    num_batches = int(np.ceil(len(train_images) / batch_size))

    # We flatten the parameters (transform the lists or tupples into numpy arrays)

    flattened_combined_params_init, unflat_params = flatten(combined_params_init)

    # Actual objective to optimize that receives flattened params

    def objective(flattened_combined_params):

        combined_params = unflat_params(flattened_combined_params)
        data_idx = batch
        gen_params, rec_params = combined_params

        # We binarize the data
    
        on = train_images[ data_idx ,: ] > npr.uniform(size = train_images[ data_idx ,: ].shape)
        images = train_images[ data_idx, : ] * 0.0
        images[ on ] = 1.0

        return vae_lower_bound(gen_params, rec_params, images) 

    # Get gradients of objective using autograd.

    objective_grad = grad(objective)
    flattened_current_params = flattened_combined_params_init

    # ADAM parameters
    
    t = 1

    # Initial values for the ADAM parameters (including the m and v vectors)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10**-8
    m = np.zeros_like(flattened_current_params)
    v = np.zeros_like(flattened_current_params)

    # We do the actual training

    for epoch in range(num_epochs):

        elbo_est = 0.0

        for n_batch in range(int(np.ceil(N / batch_size))):

            batch = np.arange(batch_size * n_batch, np.minimum(N, batch_size * (n_batch + 1)))
            grad = objective_grad(flattened_current_params)

            # Update the paramters using the ADAM updates with noisy gradient

            m = beta1*m + (1-beta1)*grad
            v = beta2*v + (1-beta2)*grad**2
            m_unbiased = m/(1-beta1**t)
            v_unbiased = v/(1-beta2**t)

            flattened_current_params += alpha*m_unbiased/(np.sqrt(v_unbiased)+epsilon)
            elbo_est += objective(flattened_current_params)
            
            t += 1

        print("Epoch: %d ELBO: %e" % (epoch, elbo_est / np.ceil(N / batch_size)))

    # We obtain the final trained parameters

    gen_params, rec_params = unflat_params(flattened_current_params)

    ### TASK 3.1 ###

    # Generate 25 images from prior and save
    z_prior_samples = npr.randn(25, latent_dim)
    x_samples = neural_net_predict(gen_params, z_prior_samples)
    save_images(sigmoid(x_samples), "task_3_1")


    ### TASK 3.2 ###

    # Generate image reconstructions for the first 10 test images    
    test_first_10_images = test_images[:10]
    encoder_output = neural_net_predict(rec_params, test_first_10_images)
    latent_variables_samples = sample_latent_variables_from_posterior(encoder_output)
    decoder_output = neural_net_predict(gen_params, latent_variables_samples)

    reconstruction_images = np.append(
                        test_first_10_images, 
                        sigmoid(decoder_output), 
                        axis = 0
                    )

    save_images(reconstruction_images, "task_3_2")


    ### TASK 3.3 ###

    # Generate 5 interpolations (by a convex conbination) from the 
    # first test image to the second test image, for the third to
    # the fourth and so on until 5 interpolations are computed in 
    # latent space

    num_interpolations = 5
    num_interpolation_steps = 25

    for i in range(5):

        # Get first and second image to compute interpolations
        first_image = neural_net_predict(rec_params, [test_images[2*i]])
        second_image = neural_net_predict(rec_params, [test_images[2*i+1]])

        # Use mean of the recognition model as the latent representation.        
        D = np.shape(first_image)[1] // 2
        first_latents = np.array(first_image[:, :D])
        second_latents = np.array(second_image[:, :D])

        # Get interpolation scalars using a convex conbination
        S = np.linspace(0, 1, num_interpolation_steps)[::-1]
        interpolation = np.array([s*first_latents + (1-s)*second_latents for s in S])

        # Reshape to (num_interpolation_steps, n_features)
        interpolation = interpolation.reshape(num_interpolation_steps, -1)

        # Compute in latent space and save them using save images
        image = neural_net_predict(gen_params, interpolation)
        save_images(sigmoid(image), f"task_3_3_{i}")

