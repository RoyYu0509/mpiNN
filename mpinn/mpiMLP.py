from .mlp import MLP, ActivationFunction
from .mpiDataDistribution import MPIDD
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

class mpiMLP:
    def __init__(self, model: MLP):
        self.model = model
        self.COMM = MPI.COMM_WORLD
        self.RANK = self.COMM.Get_rank()
        self.SIZE = self.COMM.Get_size()
        self.X_val = None
        self.y_val = None



    def mpiSGD(self,
               # Data args
               file_path, readin_chunksize,
               # Model training args
               valid_portion = 0.1,
               lr: float = 1e-3,
               epochs: int = 1000,
               batch_portion: float = 0.1,   # FIX: annotate as float; supports int too
               patience: int = 10,
               lr_decay: float = 0.5,
               report_per: int = 5000,
               lr_resch_stepsize: int = 5000,
               grad_clip: int = 1e2,
               save_fig: str = None
               ):
        patience_counter = 0
        best_data_loss = np.inf
        training_loss_his = []
        validation_loss_his = []

        # Distribute dataset

        Distributor = MPIDD(file_path, readin_chunksize)
        Distributor.mpi_load_in_and_distribute(target_col="total_amount", shuffle=True)
        X, y = Distributor.get_X_y(target_name="total_amount")
        n_local = X.shape[0]
        self.COMM.Barrier()
        print()
        print(f"Finished Distributing Data on proc {self.RANK}.........", flush=True)
        print(f"Rank {self.RANK} has {n_local} rows of data")
        print()

        # To numpy
        X = X.to_numpy()
        y = y.to_numpy()

        # enforce numeric dtype (prevents silent object/mixed issues)
        X = X.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)

        # RNG once (per-rank deterministic)
        rng = np.random.default_rng(self.RANK + 12345)

        n_local = X.shape[0]
        valid_size = 0 if n_local <= 1 else min(max(1, int(np.ceil(valid_portion * n_local))), n_local - 1)
        perm = rng.permutation(n_local)
        val_idx, tr_idx = perm[:valid_size], perm[valid_size:]

        self.X_val = X[val_idx].astype(np.float64, copy=False)
        self.y_val = y[val_idx].reshape(-1, 1).astype(np.float64, copy=False)
        X = X[tr_idx].astype(np.float64, copy=False)
        y = y[tr_idx].reshape(-1, 1).astype(np.float64, copy=False)
        n_local = X.shape[0]

        # Compute batch size (supports fraction or absolute integer)
        batch_size = int(np.ceil(max(1, batch_portion * n_local)))
        # Clamp to [1, n_local], and handle tiny shards
        batch_size = max(1, min(batch_size, n_local))

        # def global_percentiles(x: np.ndarray, qs=(0, 50, 90, 95, 99, 99.5, 99.9)):
        #     # Build a shared histogram per rank, then allreduce
        #     # Simple version: compute local percentiles and gather to root for inspection
        #     loc = np.percentile(x, qs)
        #     all_loc = None
        #     if self.RANK == 0:
        #         all_loc = np.empty((self.SIZE, len(qs)), dtype=np.float64)
        #     self.COMM.Gather(loc, all_loc, root=0)
        #     if self.RANK == 0:
        #         print("[DIAG] y percentiles per rank (qs", qs, "):")
        #         for r in range(self.SIZE):
        #             print(f"  rank {r}: {all_loc[r]}")
        #         # crude global: take min over ranks for low qs, max for high qs
        #         print("[DIAG] min over ranks at each q:", all_loc.min(axis=0))
        #         print("[DIAG] max over ranks at each q:", all_loc.max(axis=0))
        #
        # global_percentiles(y.ravel(), qs=(0, 50, 90, 95, 99, 99.5, 99.9))
        # global_percentiles(self.y_val.ravel(), qs=(0, 50, 90, 95, 99, 99.5, 99.9))

        # Training loop
        print()
        print(f"Training Starts on Rank {self.RANK}......")
        print()
        for epoch in range(epochs):
            # Sample a local minibatch
            if batch_size >= n_local:
                idx = np.arange(n_local)
            else:
                idx = rng.choice(n_local, size=batch_size, replace=False)
            X_bat, y_bat = X[idx], y[idx]

            # Local gradients
            grad_dict = self.model.compute_batch_grads(X_bat, y_bat)
            local_training_loss = float(grad_dict["loss"])  # FIX: scalarize for clean compare/print

            # ensure contiguous float64 for MPI
            dW1 = np.ascontiguousarray(grad_dict["dW1"], dtype=np.float64)
            db1 = np.ascontiguousarray(grad_dict["db1"], dtype=np.float64)
            dW2 = np.ascontiguousarray(grad_dict["dW2"], dtype=np.float64)
            db2 = np.ascontiguousarray(grad_dict["db2"], dtype=np.float64)

            # Allreduce (sum), then average
            self.COMM.Allreduce(MPI.IN_PLACE, dW1, op=MPI.SUM)
            self.COMM.Allreduce(MPI.IN_PLACE, db1, op=MPI.SUM)
            self.COMM.Allreduce(MPI.IN_PLACE, dW2, op=MPI.SUM)
            self.COMM.Allreduce(MPI.IN_PLACE, db2, op=MPI.SUM)
            scale = 1.0 / self.SIZE
            dW1 *= scale; db1 *= scale; dW2 *= scale; db2 *= scale

            gclip = grad_clip
            np.clip(dW1, -gclip, gclip, out=dW1)
            np.clip(db1, -gclip, gclip, out=db1)
            np.clip(dW2, -gclip, gclip, out=dW2)
            np.clip(db2, -gclip, gclip, out=db2)

            # Optional consistency check every 500 epochs
            tolerence = 1e-13
            def check_equality_across_procs(tolerence=tolerence):
                params = [self.model.W1, self.model.b1, self.model.W2, self.model.b2]
                local_vec = np.array([p.sum(dtype=np.float64) for p in params], dtype=np.float64)
                vec_max = local_vec.copy(); vec_min = local_vec.copy()
                self.COMM.Allreduce(MPI.IN_PLACE, vec_max, op=MPI.MAX)
                self.COMM.Allreduce(MPI.IN_PLACE, vec_min, op=MPI.MIN)
                assert np.allclose(vec_max, vec_min, atol=tolerence), "Parameters diverged!"
            if epoch % 500 == 0:
                check_equality_across_procs(tolerence=tolerence)

            # SGD update
            self.model.W1 -= lr * dW1
            self.model.b1 -= lr * db1
            self.model.W2 -= lr * dW2
            self.model.b2 -= lr * db2

            # LR decay each epoch (as you specified)
            if (lr_decay is not None) and (epoch % lr_resch_stepsize == 0):
                lr *= lr_decay

            # Compute the training & validation loss value on rank 0
            # All procs = None, except for rank 0
            train_loss = self.mpi_compute_MSE_root(X_bat, y_bat)

            val_loss = self.mpi_compute_MSE_root(self.X_val, self.y_val)
            losses = np.array([0.0, 0.0], dtype='d')

            # Boardcast the loss
            if self.RANK == 0:
                # Fill the buff in root proc
                losses[0] = train_loss
                losses[1] = val_loss

            req = self.COMM.Ibcast(losses, root=0)
            req.Wait()
            train_loss, val_loss = float(losses[0]), float(losses[1])

            # Logging
            training_loss_his.append(train_loss)
            validation_loss_his.append(val_loss)

            # Report the loss per 5000 iterations
            if (epoch % report_per == 0) and (self.RANK==0):
                print()
                print(f"Aggregating Losses to Rank {self.RANK}...")
                print()
                print(f"{epoch}-th Iteration Training Loss = {train_loss}")
                print(f"{epoch}-th Iteration Validation Loss = {val_loss}")

            # Check early stopping using validation loss, trivial computation
            if val_loss < best_data_loss - 1e-10:
                best_data_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print()
                print(f"Early Stopping Triggered on Rank {self.RANK} at iter {epoch}...")
                print()
                break
        print()
        print(f"Finished Training on Rank {self.RANK}...")
        print()
        
        # --- PLOT AND SAVE FIGURE ---
        if save_fig is not None:
            train_rmse = np.sqrt(training_loss_his)
            val_rmse = np.sqrt(validation_loss_his)

            plt.figure(figsize=(10, 6))
            plt.plot(train_rmse, label="Train RMSE")
            plt.plot(val_rmse, label="Validation RMSE")
            plt.xlabel("Epoch / Iteration")
            plt.ylabel("RMSE")
            plt.title("Training History: RMSE vs Iteration")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_fig, dpi=300)
            plt.close()
            print(f"Training history plot saved as '{save_fig}'")
        
        return training_loss_his, validation_loss_his

    def mpi_compute_MSE_root(self, X, y):
        y_est = self.model.forward(X)
        err = (y_est - y).ravel()
        n_local = int(err.size)
        sse_local = float(np.dot(err, err))

        sse_total = self.COMM.reduce(sse_local, op=MPI.SUM, root=0)
        n_total = self.COMM.reduce(n_local, op=MPI.SUM, root=0)

        if self.RANK == 0:
            if n_total == 0:
                raise ValueError("No samples across all ranks.")
            return sse_total / n_total

        return None
    
    def compute_RMSE(self, X_test, y_test):
        """Compute the test RMSE with mpi"""
        # Ensure column vector shape for y
        y_test = y_test if y_test.ndim == 2 else y_test.reshape(-1, 1)
        # All procs compute local MSE and agg to rank 0
        test_mse = self.mpi_compute_MSE_root(X_test, y_test)
        # Rank 0 boardcast MSE to all procs
        buf = np.array([0.0], dtype=np.float64)
        if self.RANK == 0:
            buf[0] = float(test_mse)
        self.COMM.Bcast(buf, root=0)

        return float(np.sqrt(buf[0]))