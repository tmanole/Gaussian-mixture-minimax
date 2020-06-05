import numpy as np
from functions import *
from models import *
import argparse
import matplotlib.pyplot as plt
import matplotlib
import sys

text_size = 17

matplotlib.rc('xtick', labelsize=text_size) 
matplotlib.rc('ytick', labelsize=text_size) 
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

lw = 3
elw=0.3

def plot_model(model, pi, r_loss, n0):
    fig = plt.figure()

    if model == 1:
        def get_params(n):
            return params_model1(n, pi)
    
        M = np.load("results/result_model1_pi" + str(pi)[2:] + ".npy")

    elif model == 2:
        def get_params(n):
            return params_model2(n)

        M = np.load("results/result_model2_pi5.npy")

    elif model == 3:
        def get_params(n):
            return params_model3(n)

        M = np.load("results/result_model3_pi5.npy")

    elif model == 4:
        def get_params(n):
            return params_model4(n)

        M = np.load("results/result_model4_pi5.npy")
    
    elif model == 5:
        def get_params(n):
            return params_model5(n)

        M = np.load("results/result_model5_pi5.npy")
    
    else:
        sys.exit("Model unrecognized.")
    
    if pi == 0.5: 
        metric = symloss
    
    else:
        metric = asymloss

    if pi != 0.5:   
        lab = "$\psi_" + str(r_loss) + "(\hat{\\boldsymbol{\eta}}_n, \\boldsymbol{\eta}_n)$" 
   
    else:
        lab = "$\\varphi_" + str(r_loss) + "(\hat{\\boldsymbol{\eta}}_n, \\boldsymbol{\eta}_n)$" 
    
    n_num = M.shape[0]
    ns = np.linspace(1000, 100000, n_num)
    
    loss = []
    yerr=[]

    for i in range(M.shape[0]):
        losses = []
    
        theta_true, v_true_1, v_true_2 = get_params(ns[i])
    
        for j in range(M.shape[1]):
            losses.append(metric(r_loss, M[i,j,1], theta_true, M[i,j,2], v_true_1, M[i,j,3], v_true_2))
    
        loss.append(np.sum(losses)/M.shape[1])
        yerr.append(np.std(losses))

    ns = np.linspace(1000, 100000, n_num)
    
    Y = np.array(np.log(loss)).reshape([-1,1])
    plt.errorbar(np.log(ns), Y, yerr=yerr, lw=lw, elinewidth=elw, label=lab,zorder=2)

    ll = 0    
    X = np.empty([ns.size-n0, 2])
    X[:,0] = np.repeat(1, ns.size-n0)
    X[:,1] = np.log(ns[n0:])
    Y = Y[n0:]        

    beta = (np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y) 
    print(beta)
    plt.plot(X[:,1], X @ beta, lw=lw, label="$" + str(np.round(np.exp(beta[0,0]), 2)) + "\cdot n^{" + str(np.round(beta[1,0],2)) + " }$",zorder=5 )

#    plt.ylim([-2.0, -0.2])

    plt.xlabel("$\log n$", fontsize=text_size)
    plt.ylabel("Log Loss", fontsize=text_size)#"$\log$ " + lab)
    plt.legend(loc="upper right", title="", prop={'size': text_size})
 
    plt.savefig("results/plots/plot_model" + str(model) + "_pi" + str(pi)[2:] + ".pdf", bbox_inches = 'tight',pad_inches = 0)

plot_model(1, 0.1, 6, 19) 
plot_model(1, 0.25, 6,19) 
plot_model(1, 0.4, 6, 19) 
plot_model(1, 0.5, 4, 0) 
plot_model(2, 0.5, 4, 0) 
plot_model(3, 0.5, 4,0) 
plot_model(4, 0.5, 4,0) 
plot_model(5, 0.5, 4,0) 

