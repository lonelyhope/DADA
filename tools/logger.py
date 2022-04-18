import os
from tensorboardX import SummaryWriter

class Logger():
    # tensorboard --logdir root
    def __init__(self, root):
        self.root = root
        self.writer = SummaryWriter(root)
    
    def path(self):
        return self.root

    def add_scalar(self, value, n_iter, name='loss'):
        # print('add', name, value, n_iter)
        self.writer.add_scalar('data/'+str(name), value, n_iter)

    def close(self):
        self.writer.close()