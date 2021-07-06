import numpy as np

class Completer:
    def __init__(self, i, T, logger, directions):
        self.T = T
        self.id = i
        self.H = None
        self.C = None
        self.X = None # debug
        self.M = None # debug
        self.ids = [] # entries are node id for each col
        self.directions = directions
        self.penalties = None # debug
        self.logger = logger

    def store_raw_table(self, X, M, none_M, broads, X_abs, M_abs, none_M_abs, ids_abs):
        self.X = X.copy()
        self.M = M.copy()
        self.none_M = none_M.copy()
        self.logger.write_str('\t\tMC')
        # print(X)
        # print(M)

        # print(broads)
        penalties = [str(round(i,2)) for i in self.penalties]

        if np.isnan(np.sum(self.H)):
            print(self.id, 'return NAN')
            print(self.H)
            sys.exit(1)

        argmins = np.argmin(self.H, axis=1)
        min_nodes = [self.ids[i] for i in argmins]

        self.logger.write_str('penalties: '+ ' '.join(penalties))
        self.logger.log_mats([
                     self.logger.format_double_masked_mat(self.X, self.M, self.none_M, self.ids, False),
                     self.logger.format_array(broads, 'src'),
                     self.logger.format_array(min_nodes, 'min'),
                     self.logger.format_mat(self.H, self.ids, False),
                     self.logger.format_double_masked_mat(X_abs, M_abs, none_M_abs, ids_abs, False)])


    def store_HC(self, H, C, node_ids, penalties):
        self.H = H.copy()
        self.C = C.copy()
        self.ids = node_ids.copy()
        self.penalties = penalties.copy()











    
