import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit

from util_funcs import sigmoid, sigmoid_gradient, tanh, tanh_gradient, get_Batch


def init_random(dx, dm, dz, option = "xavier"):
    """
    parameter initialization
    xavier initialization for weights
    all zero for bias
    can be used to initialize all zero variables for ADAM by setting "option = zeros"
    """
    
    # weights initialization
    if option == "zeros":
        # only for variables in ADAM algorithm, not to be used for true model parameters
        q_W1 = np.zeros((dm, dx))
        p_W5 = np.zeros((dx, dm))
        q_W2 = np.zeros((dz, dm))
        q_W3 = np.zeros((dz, dm))
        p_W4 = np.zeros((dm, dz))
    elif option == "xavier":
        bound = np.sqrt(6)/ np.sqrt(dx + dm)
        q_W1 = np.random.uniform(-bound, bound, (dm, dx))
        p_W5 = np.random.uniform(-bound, bound, (dx, dm))
        bound = np.sqrt(6)/ np.sqrt(dm + dz)
        q_W2 = np.random.uniform(-bound, bound, (dz, dm))
        q_W3 = np.random.uniform(-bound, bound, (dz, dm))
        p_W4 = np.random.uniform(-bound, bound, (dm, dz))
    
    # bias initialization
    q_b1 = np.zeros((dm, 1))
    p_b5 = np.zeros((dx, 1))
    q_b2 = np.zeros((dz, 1))
    q_b3 = np.zeros((dz, 1))
    p_b4 = np.zeros((dm, 1))
    
    W = [q_W1, q_W2, q_W3, p_W4, p_W5]
    b = [q_b1, q_b2, q_b3, p_b4, p_b5]
    
    return W, b


def encoder_forward_vec(X, W, b):
    """
    encoder forward propagation - vectorized version
    X: M by dx
    """
    
    q_W1, q_W2, q_W3, d,d = W
    q_b1, q_b2, q_b3, d,d = b
    
    q_a1 = q_W1 @ X.T + q_b1 # dm by M
    q_h1 = tanh_nb(q_a1) # dm by M
    q_mu = q_W2 @ q_h1 + q_b2 # dz by M
    q_a2 = q_W3 @ q_h1 + q_b3 # dz by M
    q_s2 = np.exp(q_a2) # dz by M
    
    return q_h1.T, q_a2.T, q_mu.T, q_s2.T


def sample_z_vec(q_mu, q_s2, eps):
    """sample latent variable z - vectorized version"""
    
    M, dz = q_mu.shape 

    return q_mu.reshape(M,1,dz) + np.sqrt(q_s2).reshape(M,1,dz) * eps


def decoder_forward_vec(W, b, z):
    """
    decoder forward propagation - vectorized version
    z: M by L by dz
    """
    
    d,d,d, p_W4, p_W5 = W
    d,d,d, p_b4, p_b5 = b
    
    p_a3 = z @ p_W4.T + p_b4.T # M by L by dm
    p_h2 = tanh_nb(p_a3) # M by L by dm
            
    p_a4 = p_h2 @ p_W5.T + p_b5.T # M by L by dx
    y = sigmoid_nb(p_a4) # M by L by dx
    
    return y, p_h2


def total_loss_vec(X, y, q_a2, q_mu, q_s2):
    """target total loss function - to minimize - vectorized version"""
    
    M, dx, L = X.shape[0], X.shape[1], y.shape[1]
    
    # reconstruction loss for each sample of latent variable
    loss = -np.sum(X.reshape(M,1,dx) * np.log(y) + (1-X.reshape(M,1,dx))* np.log(1-y)) / L
    
    # KL divergence/ regularization
    loss = (loss + np.sum(np.power(q_mu,2) + q_s2 - q_a2 - 1)/2)/M
    
    return loss


def para_update(W, b, dW, db, alpha):
    """update weights and bias for gradient descent"""
    
    assert len(W) == len(b) == len(dW) == len(db)
    n = len(W)
    
    for i in range(n):
        W[i] = W[i] - alpha * dW[i]
        b[i] = b[i] - alpha * db[i]
    
    return W, b


# jit decorator for above functions
sigmoid_nb = jit(sigmoid, nopython=True, cache=True)
sigmoid_gradient_nb = jit(sigmoid_gradient, nopython=True, cache=True)
tanh_nb = jit(tanh, nopython=True, cache=True)
tanh_gradient_nb = jit(tanh_gradient, nopython=True, cache=True)
get_Batch_nb = jit(get_Batch, nopython=True)
init_random_nb1 = jit(init_random, nopython=True)
sample_z_vec_nb = jit(sample_z_vec, nopython=True, cache=True)
encoder_forward_vec_nb1 = jit(encoder_forward_vec, nopython=True, cache=True)
decoder_forward_vec_nb1 = jit(decoder_forward_vec, cache=True)
total_loss_vec_nb = jit(total_loss_vec, nopython=True, cache=True)
para_update_nb = jit(para_update, nopython=True, cache=True)


@jit(cache=True)
def batch_forward_vec_nb2(Spec, X, W, b, eps):
    """forward propagation for one mini batch - full vectorized version"""
    d, M, L, d, dx, dm, dz, d, d = Spec

    p_h2 = np.zeros((M, L, dm))
    z = np.zeros((M, L, dz))
    y = np.zeros((M, L, dx))
    
    q_h1, q_a2, q_mu, q_s2 = encoder_forward_vec_nb1(X, W, b)
    z = sample_z_vec_nb(q_mu, q_s2, eps)
    y, p_h2 = decoder_forward_vec_nb1(W, b, z)
    
    loss = total_loss_vec_nb(X, y, q_a2, q_mu, q_s2)

    return y, q_h1, p_h2, q_mu, q_s2, z, loss


@njit
def grad_vec2_nb2(X, y, q_W1, q_W2, q_W3, p_W4, p_W5, q_b1, q_b2, q_b3, p_b4, p_b5, q_h1, p_h2, q_mu, q_s2, z, eps):
    """"
    batch gradient calculation - vectorized version
    not using lists for parameters to enable numba nopython
    
    inputs:
        X: Data [M by dx]
        y: Model results [M by L by dx]
        q_W1, q_W2, q_W3: Weights for encoder
        p_W4, p_W5: Weights for decoder
        q_b1, q_b2, q_b3: Bias for encoder
        p_b4, p_b5: Bias for decoder
        q_h1 (M by dm), p_h2 (M by L by dm): intermediate activation variables
        eps (L by dz), z (M by L by dz), q_s2 (M by dz), q_mu (M by dz): for sampling latent variables from posterior
    """
    
    M = X.shape[0]
    L = y.shape[1]
    
    # initialize gradient variables
    
    # L: loss; R: regularization; J: total target
    dL_dW1 = dJ_dW1 = dR_dW1 = np.zeros_like(q_W1)
    dL_db1 = dJ_db1 = dR_db1 = np.zeros_like(q_b1)
    dL_dW2 = dJ_dW2 = dR_dW2 = np.zeros_like(q_W2)
    dL_db2 = dJ_db2 = dR_db2 = np.zeros_like(q_b2)
    dL_dW3 = dJ_dW3 = dR_dW3 = np.zeros_like(q_W3)
    dL_db3 = dJ_db3 = dR_db3 = np.zeros_like(q_b3)
    dL_dW4 = dJ_dW4 = np.zeros_like(p_W4)
    dL_db4 = dJ_db4 = np.zeros_like(p_b4)    
    dL_dW5 = dJ_dW5 = np.zeros_like(p_W5)
    dL_db5 = dJ_db5 = np.zeros_like(p_b5)

    # back propagation for loss
    for iL in range(L):
        y_iL = y[:,iL,:] # M by dx
        p_h2_iL = p_h2[:,iL,:] # M by dm
        z_iL = z[:,iL,:] # M by dz
        
        L_d4 = y_iL - X # M by dx
        dL_dW5 = dL_dW5 + L_d4.T @ p_h2_iL # dx by dm
        dL_db5 = dL_db5 + np.sum(L_d4, axis = 0).reshape(-1,1) # dx by 1
        
        L_d3 = L_d4 @ p_W5 * tanh_gradient_nb(p_h2_iL) # M by dm
        dL_dW4 = dL_dW4 + L_d3.T @ z_iL # dm by dz
        dL_db4 = dL_db4 + np.sum(L_d3, axis = 0).reshape(-1,1) # dm by 1
        
        L_d22 = L_d3 @ p_W4 * eps[iL,] * np.sqrt(q_s2) / 2  # M by dz
        dL_dW3 = dL_dW3 + L_d22.T @ q_h1 # dz by dm
        dL_db3 = dL_db3 + np.sum(L_d22, axis = 0).reshape(-1,1) # dz by 1
        
        L_d21 = L_d3 @ p_W4 # M by dz
        dL_dW2 = dL_dW2 + L_d21.T @ q_h1 # dz by dm
        dL_db2 = dL_db2 + np.sum(L_d21, axis = 0).reshape(-1,1) # dz by 1

        L_d1 = (L_d21 @ q_W2 + L_d22 @ q_W3) * tanh_gradient_nb(q_h1) # M by dm
        dL_dW1 = dL_dW1 + L_d1.T @ X # dm by dx
        dL_db1 = dL_db1 + np.sum(L_d1, axis = 0).reshape(-1,1) # dm by 1

    # back propagation for regularization
    R_d22 = (q_s2 - 1)/2 # M by dz
    dR_dW3 = dR_dW3 + R_d22.T @ q_h1 # dz by dm
    dR_db3 = dR_db3 + np.sum(R_d22, axis = 0).reshape(-1,1) # dz by 1
    
    R_d21 = q_mu # M by dz
    dR_dW2 = dR_dW2 + R_d21.T @ q_h1 # dz by dm
    dR_db2 = dR_db2 + np.sum(R_d21, axis = 0).reshape(-1,1) # dm by 1

    R_d1 = (R_d22 @ q_W3 + R_d21 @ q_W2) * tanh_gradient_nb(q_h1) # M by dm
    dR_dW1 = dR_dW1 + R_d1.T @ X # dm by dx
    dR_db1 = dR_db1 + np.sum(R_d1, axis = 0).reshape(-1,1) # dm by 1
    
    dJ_dW1 = dL_dW1 / L / M + dR_dW1 / M
    dJ_db1 = dL_db1 / L / M + dR_db1 / M
    dJ_dW2 = dL_dW2 / L / M + dR_dW2 / M
    dJ_db2 = dL_db2 / L / M + dR_db2 / M    
    dJ_dW3 = dL_dW3 / L / M + dR_dW3 / M
    dJ_db3 = dL_db3 / L / M + dR_db3 / M
    dJ_dW4 = dL_dW4 / L / M
    dJ_db4 = dL_db4 / L / M 
    dJ_dW5 = dL_dW5 / L / M
    dJ_db5 = dL_db5 / L / M
    
    return dJ_dW1, dJ_dW2, dJ_dW3, dJ_dW4, dJ_dW5, dJ_db1, dJ_db2, dJ_db3, dJ_db4, dJ_db5


@jit(cache=True)
def train_AEVB(trainX, trainy, nBatch, M = 100, L = 1, std_const = 255, dm = 500, dz = 3, alpha = 0.005, beta1 = 0.9, beta2 = 0.999, eps_stable = 1e-8, W = "", b = "", nP = 0):
    """
    AEVB model as described in the paper
    Diederik P Kingma, Max Welling
    Auto-Encoding Variational Bayes (2013).
    
    Training using ADAM algorithm as described in the paper
    Diederik P Kingma, Jimmy Ba
    Adam: A Method for Stochastic Optimization (2014).
    
    Input parameters：
    ----------
    trainX: array_like
            Training dataset inputs.
            Dimension: number of sample by dim ...
    trainy: array_like
            Training dataset labels. 
            This variable is not currently used in the function. For further developments.
    nBatch: integer
            Number of mini-batch to train.
    M: integer, optional
            Size of mini-batch.
            Default at 100 as recommended in the paper.
    L: integer, optional
            Number of latent variable to sample.
            Default at 1 as recommended in the paper.
    std_const: scalar, optional
            Normlizing constant for data.
            Currently default at 255 which is usually used for black and white image data.
    dm: integer, optional
            Dimension for middle layer of the encoder and decoder.
            Default at 500 which used for MNIST dataset in the paper.
    dz: integer, optional
            Dimension for latent variables. 
            Currently default at 3
    alpha: float, optional
            Learning rate.
            Default at 0.005.
    beta1: float, optional
            Parameter for ADAM.
            Default at 0.9.
    beta2: float, optional
            Parameter for ADAM.
            Default at 0.999.
    eps_stable: float, optional
            Parameter for ADAM.
            Default at 1e-08.
    W: list, optional
            List of model weights parameters, same format as function output variable W.
            In case user wants to start training from existing parameters.
    b: list, optional
            List of model bias parameters, same format as function output variable b.
            In case user wants to start training from existing parameters.
    nP: integer, optional
            If specified with non-zero number, function will print out 
            status message after completing every nP batches.
    
    Returns:
    ----------
    W: list
            List of model weights parameters in the order of: q_W1, q_W2, q_W3, p_W4, p_W5.
            q_W1, q_W2, q_W3: weights for Gaussian MLP encoder as specified in paper Appendix C.
            p_W4, p_W5: weights for Bernoulli MLP decoder as specified in paper Appendix C
    b: list
            List of model bias parameters in the order of q_b1, q_b2, q_b3, p_b4, p_b5.
            q_b1, q_b2, q_b3: bias for Gaussian MLP encoder as specified in paper Appendix C.
            p_b4, p_b5: weights for Bernoulli MLP decoder as specified in paper Appendix C.
    loss: array-like
            Array which stores total loss for each mini-batch.
    """
    
    dx = trainX.shape[1]
    Spec = [nBatch, M, L, std_const, dx, dm, dz, alpha, nP]
    
    # initiate parameters for ADAM
    # need to use separate lines of codes otherwise they share the same reference
    v_dW, v_db = init_random_nb1(dx, dm, dz, option = "zeros")
    s_dW, s_db = init_random_nb1(dx, dm, dz, option = "zeros")
    vc_dW, vc_db = init_random_nb1(dx, dm, dz, option = "zeros")
    sc_dW, sc_db = init_random_nb1(dx, dm, dz, option = "zeros")
    num_para = len(v_dW)
    
    # weights and bias initialization
    if len(W) == len(b) == 0:
        W, b = init_random_nb1(dx, dm, dz)
    
    q_W1, q_W2, q_W3, p_W4, p_W5 = W
    q_b1, q_b2, q_b3, p_b4, p_b5 = b

    # loss
    loss = np.zeros(nBatch)
    
    for iB in range(nBatch):
        # sample a random batch
        batchX, batchy = get_Batch_nb(M, trainX, trainy)
        X = batchX / std_const

        # sample random noise for latent variable, assuming each batch uses the same random noise for now
        eps = np.random.randn(L, dz)

        y, q_h1, p_h2, q_mu, q_s2, z, loss[iB] = batch_forward_vec_nb2(Spec, X, W, b, eps)
        
        dJ_dW1, dJ_dW2, dJ_dW3, dJ_dW4, dJ_dW5, dJ_db1, dJ_db2, dJ_db3, dJ_db4, dJ_db5 =        grad_vec2_nb2(X, y, q_W1, q_W2, q_W3, p_W4, p_W5, q_b1, q_b2, q_b3, p_b4, p_b5, q_h1, p_h2, q_mu, q_s2, z, eps)
        
        dW = [dJ_dW1, dJ_dW2, dJ_dW3, dJ_dW4, dJ_dW5]
        db = [dJ_db1, dJ_db2, dJ_db3, dJ_db4, dJ_db5]
        
        # ADAM
        for i in range(num_para):
            v_dW[i] = beta1*v_dW[i] + (1-beta1)*dW[i]
            v_db[i] = beta1*v_db[i] + (1-beta1)*db[i]
            s_dW[i] = beta2*s_dW[i] + (1-beta2)*np.power(dW[i],2)
            s_db[i] = beta2*s_db[i] + (1-beta2)*np.power(db[i],2)
        
            vc_dW[i] = v_dW[i]/(1-beta1**(iB+1))
            vc_db[i] = v_db[i]/(1-beta1**(iB+1))
            sc_dW[i] = s_dW[i]/(1-beta2**(iB+1))
            sc_db[i] = s_db[i]/(1-beta2**(iB+1))
        
            dW[i] = vc_dW[i] / (np.sqrt(sc_dW[i]) + eps_stable)
            db[i] = vc_db[i] / (np.sqrt(sc_db[i]) + eps_stable)
        
        W, b = para_update_nb(W, b, dW, db, alpha)
        
        if (nP != 0) and (iB+1) % nP == 0:
            print("Batch " + str(iB+1) + " completed.")

    return W, b, loss

