# Gaussian-mixture-twocomp
Code repository for reproducing the simulation study in the paper 

Manole, T., Ho, N. (2020), [Uniform Convergence Rates for Maximum Likelihood Estimation under Two-Component Gaussian Mixture Models](https://arxiv.org/abs/2006.00704), preprint, arXiv:2006.00704.

## Dependencies 
This code has only been tested using Python 3.6 and Mathematica 12.1. The dependencies for the Python code are the packages `NumPy`, `multiprocessing`, `argparse`. 

## Usage 
### Mathematica
To reproduce the results of Appendix D, run the Mathematica notebook `mathematica/asymmetric_system.nb`. 

### Python
The code implements five models 1-5, of which models 1,5,2 respectively correspond to models A, S, S' of the paper. The simulation results are saved as `NumPy` arrays under the directory `results`, and their corresponding plots are saved under the directory `results/plots`. To reproduce the plots of the paper, run 

```{python}
cd python
python plotting.py
```

To reproduce the results for any model and mixing proportion, run

```{python}
cd python
python experiment.py -m <model_number> -p <mixing_proportion>
```
