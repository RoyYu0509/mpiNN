# Parallelized Model Training 

## Parallelize the training of a simple NN model, implemented in numpy and MPI.
- Works for any number of processes.
- Distribute data evenly to all processes.
- Computing gradients and losses in parallel.
- Log the experiment results

## Try it out!
Clone this repo, install the packages and run the following command on the local computer:

### Set up environment
```
conda create -n mpipy_mpich -c conda-forge python=3.11 mpich mpi4py numpy pandas scikit-learn matplotlib
 
conda activate mpipy_mpich
```

### Experiments with different number of processes 
**Run experiments on 1, 2, 3 processes**
```
mpiexec -np 1 python -u -m experiments "['relu','sigmoid','tanh']" "[1e-7, 5e-7, 1e-6, 5e-6, 1e-5]" 1
mpiexec -np 2 python -u -m experiments "['relu','sigmoid','tanh']" "[1e-7, 5e-7, 1e-6, 5e-6, 1e-5]" 2
mpiexec -np 3 python -u -m experiments "['relu','sigmoid','tanh']" "[1e-7, 5e-7, 1e-6, 5e-6, 1e-5]" 3
```

## Note:
Use MPICH
Use Python 3.11
Training and evaluation are carried out on a laptop equipped with an Apple M4 Pro processor and 24 GB of memory.
GPT interactions for MPI development are all labeled, search for 'GPT' to see the part where we get helps.
 