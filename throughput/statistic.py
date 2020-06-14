import os
import time
import numpy as np
from thop import profile

class Throughput(object):
    """ compute thp for training iteration 
    example:
        with Throughput(model, (input0, input1, )) as tp:
            // ..... compute

        print("average throughput is: %s GFLOPS" % (tp.val()))
    """
    average_thp = [] # in GFLOPS
    def __init__(self, model, inputs):
        assert type(inputs) == type(tuple()), \
                "Type of shapes(%s) must be tuple." % (type(inputs))
        self.macs, self.params = profile(model, inputs=inputs)
        self.time_start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        compute_training = lambda mac: mac*2*3
        elapsed_time_in_sec = time.time() - self.time_start
        mac_in_gflops = compute_training(self.macs) / elapsed_time_in_sec / (1024.*1024.*1024)
        Throughput.average_thp.append(mac_in_gflops)
        if exc_type:
            print(f'exception_type: {exc_type}')
            print(f'exception_value: {exc_value}')
            print(f'exception_traceback: {exc_traceback}')

    def val(self):
        return np.mean(Throughput.average_thp)
