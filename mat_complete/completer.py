class Completer:
    def __init__(self, i, T, logger):
        self.T = T
        self.id = i
        self.H = None
        self.C = None
        self.X = None # debug
        self.M = None # debug
        self.ids = [] # entries are node id for each col
        self.logger = logger

    def store_raw_table(self, X, M, broads, X_full, M_full, ids_full):
        self.X = X.copy()
        self.M = M.copy()
        self.logger.write_str('\t\tMC')
        # print(X)
        # print(M)

        # print(broads)
        self.logger.log_mats([
                         self.logger.format_masked_mat(self.X, self.M, self.ids, False),
                         self.logger.format_array(broads, 'src'),
                         self.logger.format_mat(self.H, self.ids, False),
                         self.logger.format_masked_mat(X_full, M_full, ids_full, False)])


    def store_HC(self, H, C, node_ids):
        self.H = H.copy()
        self.C = C.copy()
        self.ids = node_ids.copy()











    
