# Parallelized Model Training using numpy and MPI.
- Works for any number of processes.
- Distribute data evenly to all processes.
- Computing gradients and losses in parallel.
- Log the experiment results
## IMPORTANT: 
**YOU NEED TO PREPARE THE FILE `nytaxi2022.csv` ON YOUR OWN.**

**Move it to the project folder so that the scripts can read it.**

# Try it out!
Clone this repo, install the packages, and run the following command on the local computer:

## Set up environment
```
conda create -n mpipy_mpich -c conda-forge python=3.11 mpich mpi4py numpy pandas scikit-learn matplotlib
 
conda activate mpipy_mpich
```

## Run sub-experiments with one activation function
```
mpiexec -np 1 python -u -m experiments "['relu']" "[240]" 1
mpiexec -np 2 python -u -m experiments "['relu']" "[240]" 2
mpiexec -np 3 python -u -m experiments "['relu']" "[240]" 3
mpiexec -np 4 python -u -m experiments "['relu']" "[240]" 4
mpiexec -np 5 python -u -m experiments "['relu']" "[240]" 5
```


## Run the full experiments on 1, 2, 3, 4processes
```
mpiexec -np 1 python -u -m experiments "['relu','sigmoid','tanh']" "[180, 240, 300, 360, 420]" 1
mpiexec -np 2 python -u -m experiments "['relu','sigmoid','tanh']" "[180, 240, 300, 360, 420]" 2
mpiexec -np 3 python -u -m experiments "['relu','sigmoid','tanh']" "[180, 240, 300, 360, 420]" 3
mpiexec -np 4 python -u -m experiments "['relu','sigmoid','tanh']" "[180, 240, 300, 360, 420]" 4
```

# Note:
- Use MPICH
- Use Python 3.11
- Training and evaluation are carried out on a laptop equipped with an Apple M4 Pro processor and 24 GB of memory.
- GPT interactions for MPI development are all labeled, search for 'GPT' to see the part where we get helps. 



