import sys


class Logger:
    def __init__(self, dirname, node_id):
        self.id = node_id
        self.filename = dirname + '/' + str(node_id)
        with open(self.filename, 'w') as writer:
            writer.write('node ' + str(node_id) + '\n')

        self.score_filename = dirname + '/score' + str(node_id)

        with open(self.score_filename, 'w') as w:
            pass

    def write_miners(self, miners):
        with open(self.score_filename, 'a') as writer:
            s = [str(a) for a in miners]
            writer.write('miners:' + ' '.join(s) + '\n')


    def write_score(self, scores):
        with open(self.score_filename, 'a') as writer:
            line = []
            for m, conn, score in scores:
                pair = str(m) + ',' + str(conn) + ','+ str(score) 
                line.append(pair)
            writer.write(' '.join(line) + '\n')



        

    def write_mat(self, A, comment):
         with open(self.filename, 'a') as writer:
            if comment is not None:
                writer.write(str(comment) + '\n')
            
            ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]
            writer.write('_'.join(ticks) + '\n')
            for i in range(A.shape[0]):
                text = ["{:5d}".format(int(a)) for a in A[i]]
                line = ' '.join(text)
                writer.write(line + '\n')

    def write_float_mat(self, A, comment, special):
         with open(self.filename, 'a') as writer:
            if comment is not None:
                writer.write(str(comment) + '\n')
            
            ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]
            writer.write('_'.join(ticks) + '\n')
            for i in range(A.shape[0]):
                text = []
                for a in A[i]:
                    if a == special: 
                        text.append("nan".rjust(5, ' '))
                    else:
                        text.append("{:5.2f}".format(a))

                line = ' '.join(text)
                writer.write(line + '\n')


    def write_conns(self, outs, ins, comment):
        with open(self.filename, 'a') as writer:
            out_txt = ["{:d}".format(int(a)) for a in outs]
            in_txt = ["{:d}".format(int(a)) for a in ins]
            line = '[O:' + ' '.join(out_txt) + ']. [I('+str(len(ins))+'):' + ' '.join(in_txt) + ' ]'
            if comment is not None:
                writer.write(str(comment) + ' ' + line + '\n' )
            else:
                writer.write(line + '\n')

    def write_ucb(self, region, arm, samples):
        with open(self.filename, 'a') as writer:
            line = 'r:'+str(region)+' a:'+str(arm)+' -> '+ str(samples)
            writer.write(line + '\n')

    def write_list(self, data, comment):
        with open(self.filename, 'a') as writer:
            text = ["{:4d}".format(int(a)) for a in data]
            line = ' '.join(text)
            if comment is not None:
                writer.write(str(comment) + line + '\n')
            else:
                writer.write(line + '\n')

    def write_str(self, txt):
        with open(self.filename, 'a') as writer:
            writer.write(txt + '\n')

    def format_mat(self, A, is_float):
        lines = []
        if not is_float:
            ticks = [str(a).rjust(4, '_') for a in range(A.shape[1])]
        else:
            ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]
        lines.append('_' + '_'.join(ticks)+'_ ')
        for i in range(A.shape[0]):
            if not is_float:
                text = ["{:4d}".format(int(a)) for a in A[i]]
            else:
                text = ["{:5.2f}".format(a) for a in A[i]]
            line = ' '.join(text)
            lines.append('[' + line + ' ]')
        return lines

    def format_array(self, a, title):
        lines = []
        lines.append('_' + ''.join(title)+'_')
        for i in range(len(a)):
            text = ["{:5s}".format(str(a[i]))]
            line = ' '.join(text)
            lines.append(' ' + line + ' ')
        return lines

    def format_score(self, scores):
        lines = []
        title = '_Rslt'
        lines.append('_' + ''.join(title)+'_')
        for i in range(len(a)):
            text = ["{:5s}".format(str(a[i]))]
            line = ' '.join(text)
            lines.append(' ' + line + ' ')
        return lines


    def format_mat_5(self, A, is_float):
        lines = []
        if not is_float:
            ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]
        else:
            ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]
        lines.append('_' + '_'.join(ticks)+'_ ')
        for i in range(A.shape[0]):
            if not is_float:
                text = ["{:5d}".format(int(a)) for a in A[i]]
            else:
                text = ["{:5.2f}".format(a) for a in A[i]]
            line = ' '.join(text)
            lines.append('[' + line + ' ]')
        return lines


    def print_mats(self, mats):
        lines = []
        num_mat = len(mats)
        num_line = len(mats[0])
        for i in range(num_line): 
            line = ''
            for mat in mats:
                line += mat[i] + '\t'
            print(line) 

    def log_mats(self, mats):
        with open(self.filename, 'a') as writer:
            lines = []
            num_mat = len(mats)
            num_line = len(mats[0])
            for i in range(num_line): 
                line = ''
                for mat in mats:
                    line += mat[i] + '\t'
                writer.write(line + '\n')

    def print_mat(self, A, is_float):
        for i in range(A.shape[0]):
            if not is_float:
                text = ["{:4d}".format(int(a)) for a in A[i]]
            else:
                text = ["{:5.2f}".format(a) for a in A[i]]
            line = ' '.join(text)
            print('[' + line + ' ]')

    def format_masked_mat(self, A, mask, is_float):
        lines = []
        if not is_float:
            ticks = [str(a).rjust(4, '_') for a in range(A.shape[1])]
        else:
            ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]

        lines.append('_' + '_'.join(ticks)+'_ ')
        for i in range(A.shape[0]):
            row = A[i]
            text = []
            if not is_float:
                for j in range(len(row)):
                    if mask[i,j] == 1:
                        text.append("{:4d}".format(int(row[j]))) 
                    elif mask[i,j] == 0:
                        text.append("{:>4}".format('*')) 
            else:
                for j in range(len(row)):
                    if mask[i,j] == 1:
                        text.append("{:5.2f}".format(row[j])) 
                    elif mask[i,j] == 0:
                        text.append("{:>5}".format('*')) 
            line = ' '.join(text)
            lines.append('[' + line + ' ]')
        return lines

