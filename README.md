# Parallelized Model Training 

## Parallelize the training of a simple NN model, implemented in numpy and MPI.
- Works for any number of processes.
- Computing gradients and losses in parallel.
- Distribute data evenly to all processes.

## Try it out!
Clone this repo and run the toy example: `mpiexec -l -np 3 python -u -m tests.testMpiTraining`

## Note:
Use Python 3.11
