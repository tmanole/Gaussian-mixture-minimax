import numpy as np
from functions import *

def params_model1(n, pi):
    c = pi/(1-pi)

    x_1 = 0 
    x_2 = 0
    y_2 = 0.1
    y_1 = -y_2/c 
    y_3 = -y_2 * (1 + (1/c))/2
 
    v_0 = 1

    epsilon_n = n**(-1.0/12)

    theta_true = epsilon_n * (x_2 - x_1)
    v_true_1 = epsilon_n**2 * y_1 + v_0 
    v_true_2 = epsilon_n**2 * (y_2 + y_3) + v_0 

    return (theta_true, v_true_1, v_true_2)

def params_model2(n):
    x_1 = 1/2.0 
    x_2 = 1/2.0
    y_1 = 1/3.0
    y_2 = 2 * x_1**2 - y_1 
    y_3 = 0
 
    v_0 = 1

    epsilon_n = n**(-1.0/8)

    theta_true = epsilon_n 
    v_true_1 = epsilon_n**2 * y_1 + v_0 
    v_true_2 = epsilon_n**2 * (y_2 + y_3) + v_0 

    return (theta_true, v_true_1, v_true_2)

def params_model3(n):
    x_1 = 1/2.0 
    x_2 = 1/2.0
    y_1 = 1/3.0
    y_2 = 2 * x_1**2 - y_1 
    y_3 = 0
 
    v_0 = 1

    epsilon_n = n**(-1.0/6)

    theta_true = epsilon_n 
    v_true_1 = epsilon_n**2 * y_1 + v_0 
    v_true_2 = epsilon_n**2 * (y_2 + y_3) + v_0 

    return (theta_true, v_true_1, v_true_2)

def params_model4(n):
    x_1 = 1/2.0 
    x_2 = 1/2.0
    y_1 = 1/3.0
    y_2 = 2 * x_1**2 - y_1 
    y_3 = 0
 
    v_0 = 1

    epsilon_n = n**(-1.0/4)

    theta_true = epsilon_n 
    v_true_1 = epsilon_n**2 * y_1 + v_0 
    v_true_2 = epsilon_n**2 * (y_2 + y_3) + v_0 

    return (theta_true, v_true_1, v_true_2)

def params_model5(n):
    x_1 = 1
    x_2 = 1.5
    y_1 = 3.5
    y_2 = 0.5 
    y_3 = -1.5
 
    v_0 = 1

    epsilon_n = n**(-1.0/8)

    theta_true = epsilon_n 
    v_true_1 = epsilon_n**2 * y_1 + v_0 
    v_true_2 = epsilon_n**2 * (y_2 + y_3) + v_0 

    return (theta_true, v_true_1, v_true_2)
