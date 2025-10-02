import csv
import pandas as pd
import pickle
from mpi4py import MPI
from .mpiTag import ACK, SAFE, DATA
from typing import List, Callable, Any
from math import floor

class MPIDD:
    """
    Limit the number of proc to less than 100,
    or change the value of `ACK`, `SAFE`, `DATA`
    """
    def __init__(self, file_path, readin_chunksize=10000):
        self.file_path = file_path
        self.COMM = MPI.COMM_WORLD
        self.RANK = self.COMM.Get_rank()
        self.SIZE = self.COMM.Get_size()
        self.READINCHUNKSIZE = readin_chunksize
        self.local_data = None

        # Data Distribution Parameters
        proc_num = self.SIZE
        self.SUBCHUNKSIZE = floor(readin_chunksize / proc_num)
        self.send_acked = [0] * proc_num  # 1 when ACK received for the df sent to proc i
        self.pending_reqs = []  # MPI.Request objects from Isend/isend
        self.everyone_safe = [0] * proc_num  # mark SAFE received from each rank

    def mpi_recieve_msg(self, status: MPI.Status):
        """
        After getting status using Iprobe, initate recieve operation
        `send_acked` & `everyone_safe` need to be modified

        GPT: "how to prepared buffer for pandas dataframe"
        """
        src = status.Get_source()
        tag = status.Get_tag()

        send_acked = self.send_acked
        everyone_safe = self.everyone_safe

        if tag == ACK:
            # Control: ACK carries a tiny Python int payload = word index i
            # We use the HIGH-LEVEL API for small Python objects (convenient).
            idx = self.COMM.recv(source=src, tag=ACK)  # -> int i
            send_acked[idx] = 1
            return

        if tag == SAFE:
            # Control: SAFE is an empty "I'm done" notice
            _ = self.COMM.recv(source=src, tag=SAFE)  # payload unused; just clear it
            everyone_safe[src] = 1
            return

        # Otherwise, it's a DATA message
        # Asking the size of the pandas data
        if tag == DATA:
            sub_chunk = self.COMM.recv(source=src, tag=DATA)
            if self.local_data is None or (
                    isinstance(self.local_data, pd.DataFrame) and self.local_data.empty):  # ← add empty check
                self.local_data = sub_chunk.copy()
            else:
                self.local_data = pd.concat([self.local_data, sub_chunk], ignore_index=True)
            self.pending_reqs.append(self.COMM.isend(self.RANK, dest=src, tag=ACK))

    def service_incoming(self):
        """Drain the network: while there are messages, receive exactly one each time."""
        while True:
            # Iprobe(...) = Nonblocking "is there a message for me?"
            # Returns (flag, status). If flag=True, 'status' is valid to use.
            status = MPI.Status()
            flag = self.COMM.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if not flag:
                break
            self.mpi_recieve_msg(status)

    def mpi_distribute_from_one_to_all(self, chunk):
        """Distribute to all processes evenly"""
        proc_num = self.SIZE
        sub_chunk_size = self.SUBCHUNKSIZE

        # Reset state on all ranks
        self.send_acked = [0] * proc_num  # 1 when ACK received for the df sent to proc i
        self.pending_reqs = []  # MPI.Request objects from Isend/isend
        self.everyone_safe = [0] * proc_num  # mark SAFE received from each rank
        # Send sub-chunk to each process
        start = 0
        end = 0

        # Sender proc
        if self.RANK == 0:
            total_rows = len(chunk)
            for i in range(proc_num):
                # Slice the portion evenly that belongs to each process
                # if p_num = 2, sub_chunk_size= 500, chunk_size=1000:
                #   start_p0=0, end_p0=500  |  start_p1=500, end_p0=1000
                start = i * sub_chunk_size
                # Let the last rank grab the remainings to avoid out of range
                if i == proc_num - 1:
                    end = total_rows
                else:
                    end = start + sub_chunk_size
                temp_subchunk = chunk.iloc[start:end]
                # Send out the sub-chunk out, tag it same as the destination
                req = self.COMM.isend(temp_subchunk, dest=i, tag=DATA)
                self.pending_reqs.append(req)
        # Reciever proc
        else:
            # Recieving from the root, (Ends this round when hearing SAFE from the root)
            while self.everyone_safe[0] == 0:
                self.service_incoming()

            # Sending SAFE when we hear from root
            req = self.COMM.send(self.RANK, dest=0, tag=SAFE)
            self.pending_reqs.append(req)

        # Synchronization:
        if self.RANK == 0:
            #   Step 1: root procs will wait for ACKs (keep draining the network so progress happens)
            #   Loop while there are some sents are not acknowledged yet; here is the
            while not all(self.send_acked):
                self.service_incoming()

            #   Step 2: All send are ack,
            #   announce SAFE to others (tiny high-level isend with tag=SAFE)
            for p in range(self.SIZE):
                # Set itself in the local indicator array to safe
                if p == self.RANK:
                    self.everyone_safe[p] = 1
                    continue
                # Set itself in all other indicator array to safe
                self.pending_reqs.append(self.COMM.isend(None, dest=p, tag=SAFE))

            #   Step 3: Wait until all neighbours of p0 tell me they are safe (ie.)
            #   Ensure I have sent all sub-chunks out to all nodes
            while not all(self.everyone_safe):
                self.service_incoming()

        #   Make sure every outstanding request has finished on both root & non-root procs
        #   前面只检查了 operations 的 logical confirmation, 这里检查的是真正的传输和接收工作是否完成
        MPI.Request.Waitall([r for r in self.pending_reqs if isinstance(r, MPI.Request)])

        # FIX: guard when local_data may still be None
        # rows = 0 if self.local_data is None else self.local_data.shape[0]
        # print(f"[Rank {self.RANK}] have received {rows} rows of data")


    def mpi_load_in_and_distribute(self, target_col="total_amount", shuffle=True):
        # Read in chunk by chunk
        """
                On proc 0 DO:
                    - Load a CSV file row by row (or chunk by chunk) using a while loop
                    - Distribute chunks evenly.
                GPT: "add shuffle"
                """
        print()
        print(f"Start Distributing Data on proc {self.RANK}.........")
        print()

        chunk = None
        has_chunk = None
        reader = None

        if self.RANK == 0:
            chunksize = self.READINCHUNKSIZE
            reader = pd.read_csv(self.file_path, chunksize=chunksize)
            root_p_counter = 1

        while True:
            if self.RANK == 0:
                try:
                    chunk = next(reader)  # get the next chunk
                    has_chunk = True
                except StopIteration:
                    chunk = None
                    has_chunk = False

            # Determine if root proc has anything to send, if no, stop the while loop
            has_chunk = self.COMM.bcast(has_chunk, root=0)
            if not has_chunk: # Every procs recieve beak has_chunk == false
                break

            # Sanity Check Before Distribution
            if self.RANK == 0:
                # Shuffle the data chunk before distribution
                if shuffle:
                    chunk = chunk.sample(frac=1).reset_index(drop=True)
                # Create the local data frame to recieve future sub-chunks at the first loading round
                if self.local_data is None:
                    self.local_data = pd.DataFrame(columns=chunk.columns)
                # Check target column name is valid
                if target_col not in chunk.columns:
                    raise ValueError(f"Target column '{target_col}' not found in CSV")

            # ALL Procs enter Distribution Round
            self.mpi_distribute_from_one_to_all(chunk)

            ## Root finished processing this chunk, report
            # if self.RANK == 0:
            #     print(f"Process {self.RANK} finished {root_p_counter}th loading round")
            #     root_p_counter += 1
        print()
        print(f"Rank {self.RANK} stores {self.local_data.shape[0]} rows.........")
        print()
        return


    def get_X_y(self, target_name="total_amount"):
        """Get the local X and y"""
        y = self.local_data[target_name]  # Series (1D)
        X = self.local_data.drop(columns=[target_name])  # DataFrame (all other columns)
        return X, y