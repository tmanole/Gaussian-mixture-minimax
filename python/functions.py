import numpy as np

def sample_mixture(theta, v_1, v_2, pi, n):
    """ Sample from the mixture. """
    c = pi/(1-pi)

    x = np.empty(n)

    for i in range(n):
        u = np.random.uniform(size=1)

        if u < pi:
            x[i] = np.random.normal(loc=-theta, scale=np.sqrt(v_1), size=1)

        else:
            x[i] = np.random.normal(loc=c*theta, scale=np.sqrt(v_2), size=1)

    return x


def density(y, mu, v):
    """ Gaussian density, without normalizing constant. """
    return np.exp(-(y-mu)**2/(2*v))

def likelihood(Y, mu, v_1, v_2, pi):
    """ Likelihood function, up to normalization. """
    return np.sum(pi * density(Y, -mu, v_1) + (1-pi) * density(Y, pi*mu/(1-pi), v_2))

def em(Y, pi, mu_start, v_start_1, v_start_2, max_iter=2000, eps=1e-8):
    """ EM Algorithm. """
    c = pi/(1-pi)

    n = Y.size

    mu_new = mu_start
    v_new_1 = v_start_1
    v_new_2 = v_start_2    

    for j in range(max_iter):
    
        denom = pi * density(Y, -mu_new, v_new_1) + (1-pi) * density(Y, c*mu_new, v_new_2)
        weights = pi * density(Y, -mu_new, v_new_1) / denom
    
        w_sum = np.sum(weights)
    
        mu_old = mu_new
        mu_new = np.sum(-weights * Y + c * (1-weights) * Y)/(w_sum + c * (n-w_sum))
    
        v_old_1 = v_new_1
        v_old_2 = v_new_2
    
        v_new_1 = np.sum(weights * (Y + mu_new)**2)/w_sum
        v_new_2 = np.sum((1-weights) * (Y- c * mu_new)**2)/(n-w_sum)

        lik = likelihood(Y, mu_new, v_new_1, v_new_2, pi)
        stopping_criterion = np.abs(likelihood(Y, mu_old, v_old_1, v_old_2, pi) - lik)
    
        if stopping_criterion < eps:
            break

    return (mu_new, v_new_1, v_new_2, lik, j)


def asymloss(r, theta_1, theta_2, v_1_first, v_2_first, v_1_second, v_2_second):
    """ Asymmetric loss function. """
    return (np.abs(theta_1 - theta_2)**r + np.abs(v_1_first - v_1_second)**(r/2) + np.abs(v_2_first - v_2_second)**(r/2))**(1.0/r)


def symloss(r, theta_1, theta_2, v_1_first, v_2_first, v_1_second, v_2_second):
    """ Symmetric loss function. """
    return (np.min([np.abs(theta_1 - theta_2)**r + np.abs(v_1_first - v_1_second)**(r/2) + np.abs(v_2_first - v_2_second)**(r/2),
                   np.abs(theta_1 - theta_2)**r + np.abs(v_1_first - v_2_second)**(r/2) + np.abs(v_2_first - v_1_second)**(r/2)]))**(1.0/r)
