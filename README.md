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

# Run Experiment
Use the following command to run experiment with the interested hyperparameters.
```
mpiexec -np {NUMBER OF PROCESSES} python -u -m experiments "{LIST OF ACT FUNC NAME}" "{LIST OF BATCH SIZE}" {NUMBER OF PROCESSES} {READ IN DATA CHUNK SIZE}
```

## Run sub-experiments with one activation function
```
mpiexec -np 1 python -u -m experiments "['relu']" "[1440]" 1 40000
mpiexec -np 2 python -u -m experiments "['relu']" "[1440]" 2 40000
```


## Run the full experiments on 1, 2, 3processes
```
mpiexec -np 1 python -u -m experiments "['relu','sigmoid','tanh']" "[480, 960, 1440, 1920, 2400]" 1 400000
mpiexec -np 2 python -u -m experiments "['relu','sigmoid','tanh']" "[480, 960, 1440, 1920, 2400]" 2 400000
mpiexec -np 3 python -u -m experiments "['relu','sigmoid','tanh']" "[480, 960, 1440, 1920, 2400]" 3 400000
```

# Note:
- Use MPICH
- Use Python 3.11
- Training and evaluation are carried out on a laptop equipped with an Apple M4 Pro processor and 24 GB of memory.
- GPT interactions for MPI development are all labeled, search for 'GPT' to see the part where we get helps. 


