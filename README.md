# Parallelized Model Training 

## Parallelize the training of a simple NN model, implemented in numpy and MPI.
- Works for any number of processes.
- Distribute data evenly to all processes.
- Computing gradients and losses in parallel.
- Log the experiment results

## Try it out!
Clone this repo, install the packages and run the following command on the local computer:
- **1 processes experiments** mpiexec -np 1 python -u -m experiments "['relu','sigmoid','tanh']"  "[5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5]" 1
- **2 processes experiments** mpiexec -np 2 python -u -m experiments "['relu','sigmoid','tanh']"  "[5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5]" 2
- **3 processes experiments** mpiexec -np 3 python -u -m experiments "['relu','sigmoid','tanh']"  "[5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5]" 3

## Note:
Use MPICH
Use Python 3.11
Training and evaluation are carried out on a laptop equipped with an Apple M4 Pro processor and 24 GB of memory.