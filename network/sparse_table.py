import numpy as np

class SparseTable:
    def __init__(self, node_id):
        self.table = []
        self.abs_table = []
        self.id = node_id

        self.broads = [] # for debug

    def append_broads(self, broads):
        self.broads += broads

    # is_abs -> is slot data is absolute time
    def append_time(self, slots, num_msg, data_type):
        lines = [[] for _ in range(num_msg)] 
        i = 0
        for p, t_list in slots.items():
            if len(t_list) != num_msg:
                print('Error. append_time sparse table')
                print(len(t_list))
                print(num_msg)
                sys.exit(2)
            # debug
            # if None in t_list:
                # print(t_list)
            for i in range(num_msg):
                t, direction = t_list[i]
                lines[i].append((p, t, direction)) 
        for i in range(num_msg):
            if data_type == 'abs_time':
                self.abs_table.append(lines[i])
            elif data_type == 'rel_time':
                self.table.append(lines[i])
            else:
                print('Unknown type in SparseTable')
                sys.exit(1)
