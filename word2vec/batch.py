from __future__ import print_function
import collections
import numpy as np

class ContextBatchGenerator:
    
    def __init__(self, text, window):
        self._text = text # -> ["1", "2", "3", "4", "5", "6", "7"]
        self._len = len(text)
        self._window = window # -> 2
        self._cursor = 0
        self._span = window*2 + 1 # -> 5 [window... , target, window...]
        self._non_window_idx = [i for i in range(self._span) if i != window] # -> [0,1,3,4]
        self._buffer = collections.deque(maxlen=self._span)
        for i in range(self._span):
            self.shift_buffer()
        # -> buffer = ["1", "2", "3", "4", "5"]
    
    def _batch(self, size):
        l = list()
        for i in range(size):
            target = self._buffer[self._window] # -> buffer[2]
            context = [self._buffer[i] for i in self._non_window_idx] # buffer ['1','2','4','5']
            l.append((context, target)) # ->(['1', '2', '4', '5'], '3')
            self.shift_buffer()
        return l
            
    def shift_buffer(self):
        self._buffer.append(self._text[self._cursor])
        self._cursor = (self._cursor + 1) % self._len

class SkipGramGenerator(ContextBatchGenerator):
    
    def next(self, size, dtype=np.int32):
        if (size % (self._window*2) !=0):
            raise ValueError("batch size should be devidable by window*2")
        batches = size // (self._window*2)
        
        batch = np.ndarray(shape=(size), dtype=dtype)
        labels = np.ndarray(shape=(size, 1), dtype=dtype)
        i = 0
        for b in self._batch(batches):
            for t in b[0]:
                batch[i] = t
                labels[i] = b[1]
                i+=1
        return batch, labels # next(2) -> [10, 30], [[20], [20]]
    
class CBOWGenerator(ContextBatchGenerator):
    
    def next(self, size, dtype=np.int32):
        if (size % (self._window*2) !=0):
            raise ValueError("batch size should be devidable by window*2")
        batches = size // (self._window*2)
        
        batch = np.ndarray(shape=(size), dtype=dtype)
        labels = np.ndarray(shape=(size, 1), dtype=dtype)
        i = 0
        for b in self._batch(batches):
            for t in b[0]:
                batch[i] = b[1] # <--
                labels[i] = t # <--
                i+=1
        return batch, labels
        

#g = SkipGramGenerator([10, 20, 30, 40, 50, 60, 70], 1)
#g.next(2) # -> [10, 30], [[20], [20]]

#g = CBOWGenerator([10, 20, 30, 40, 50, 60, 70], 1)
#g.next(2) # -> [20, 20], [[10], [30]]
