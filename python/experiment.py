import numpy as np
from functions import *
from models import *
import sys
import multiprocessing as mp
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pi', default=0.1, type=float, help='Mixing proportion.')
parser.add_argument('-m', '--model', default=1, type=int, help='Model number.')
parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.')
parser.add_argument('-r' ,'--reps', default=10, type=int, help='Number of replications per sample size.')
parser.add_argument('-mi','--maxit', default=2000, type=int, help='Maximum EM iterations.')
parser.add_argument('-e', '--eps', default=1e-8, type=float, help='EM stopping criterion.')
args = parser.parse_args()

print("Args: ", args)

pi = args.pi                          # Mixing proportion.
c = pi/(1-pi)                         # Component size ratio.
model = args.model                    # Model number.
n_proc = args.nproc                   # Number of cores to use.
max_iter = args.maxit                 # Maximum EM iterations.
eps = args.eps                        # EM Stopping criterion.
num_init = 5                          # Number of random EM initializations.
n_num = 100                           # Number of sample sizes to use.
ns = np.linspace(1000, 100000, n_num) # Sample sizes to use.
reps = args.reps                      # Number of replications to run per sample size.
init_neigh = -1/14.0                  # Starting value initialization neighborhood.

print("Chose Model " + str(model) + " with pi=" + str(pi))

do_flip = False

if model == 1:    
    def get_params(n):
        return params_model1(n, pi)

elif model == 2:
    if pi != 0.5:
        print("Warning: Running Model 2 without pi = 0.5.")

    def get_params(n):
        return params_model2(n)

elif model == 3:
    if pi != 0.5:
        print("Warning: Running Model 3 without pi = 0.5.")

    def get_params(n):
        return params_model3(n)

elif model == 4:
    if pi != 0.5:
        print("Warning: Running Model 4 without pi = 0.5.")

    def get_params(n):
        return params_model4(n)

else:
    sys.exit("Model unrecognized.")


def sample(n):
    """ Sample from the mixture. """
    th, v_1, v_2 = get_params(n)
    return sample_mixture(th, v_1, v_2, pi, n)

def init_params(n, do_flip=False):
    """ Starting values for EM algorithm. """
    theta_true, v_true_1, v_true_2 = get_params(n)

    theta_start = np.random.uniform(theta_true-n**(init_neigh), theta_true+n**(init_neigh), 1)[0]

    if do_flip:
        u = np.random.uniform(0, 1, size=1)

        if u < 0.5:
            theta_start *= -1

    v_start_1   = np.random.uniform(v_true_1-n**(2*init_neigh), v_true_1+n**(2*init_neigh), 1)[0]
    v_start_2   = np.random.uniform(v_true_2-n**(2*init_neigh), v_true_2+n**(2*init_neigh), 1)[0]

    return (theta_start, v_start_1, v_start_2)
        
def process_chunk(bound):
    """ Run EM on a range of sample sizes. """
    ind_low = bound[0]
    ind_high= bound[1]

    m = ind_high - ind_low

    seed_ctr = 2000 * ind_low   # Random seed

    chunk_result = np.empty((m, reps, 6))
    run_out = np.empty((num_init, 5))

    for i in range(ind_low, ind_high):
        n = int(ns[i])
        print(n, i)
        for rep in range(reps):
            np.random.seed(rep)
            Y = sample(n)
    
            for j in range(num_init):
                np.random.seed(seed_ctr)
                seed_ctr += 1
    
                theta_start, v_start_1, v_start_2 = init_params(n, do_flip=do_flip)
    
                run_out[j,:] = em(Y, pi, theta_start, v_start_1, v_start_2, max_iter=max_iter, eps=eps)
    
            best_ind = np.argmax(run_out[:,3])

            chunk_result[i-ind_low, rep, 1:] = run_out[best_ind,:]
            chunk_result[i-ind_low, rep, 0] = n
        
    return chunk_result

proc_chunks = []

Del = n_num // n_proc 

for i in range(n_proc):
    if i == n_proc-1:
        proc_chunks.append(( (n_proc-1) * Del, n_num) )

    else:
        proc_chunks.append(( (i*Del, (i+1)*Del ) ))

if n_proc == 8:
    # Hard code sample size chunks when using 8 cores.
    proc_chunks = [(0, 30), (30, 50), (50, 60), (60, 70), (70, 80), (80, 88), (88, 95), (95, 100)]

with mp.Pool(processes=n_proc) as pool:
    proc_results = [pool.apply_async(process_chunk,
                                     args=(chunk,))
                    for chunk in proc_chunks]

    result_chunks = [r.get() for r in proc_results]

done = np.concatenate(result_chunks, axis=0)

print(done)

np.save("results/result_model" + str(model) + "_pi" + str(pi)[2:] + ".npy", done)
