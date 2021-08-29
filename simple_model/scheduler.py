import sys

class NodeScheduler:
    def __init__(self, node_id, update_interval, num_topo):
        self.id = node_id
        self.update_interval = update_interval
        self.num_topo = num_topo
        self.init_epoch = None

        self.state = None

    def initiate_update(self, e):
        # if last update is not finished, ignore new requests
        if self.init_epoch is None:
            self.init_epoch = e

    def get_curr_state(self, e):
        if self.init_epoch is None:
            self.state ='idle' 
            return 'idle' 
        if e < self.num_topo-1:
            return 'acc'

        interval = e - self.init_epoch
        assert(interval >= 0)
        if interval+1 >= self.update_interval:
            self.init_epoch = None
            self.state ='run' 
            return 'run'
        elif interval > self.num_topo:
            print('Error. interval > acc num topolopy', interval, self.update_interval)
            sys.exit(1)
        else:
            self.state ='acc' 
            return 'acc'
