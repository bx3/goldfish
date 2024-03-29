import sys
import os

class Logger:
    def __init__(self, dirname, node_id, is_on):
        self.id = node_id
        self.filename = os.path.join(dirname, str(node_id))
        with open(self.filename, 'w') as writer:
            writer.write('node ' + str(node_id) + '\n')

        self.score_filename = dirname + '/score' + str(node_id)

        self.is_on = is_on

        with open(self.score_filename, 'w') as w:
            pass

    def write_miners(self, miners):
        if not self.is_on:
            return 
        with open(self.score_filename, 'a') as writer:
            s = [str(a) for a in miners]
            writer.write('miners:' + ' '.join(s) + '\n')


    def write_score(self, scores):
        if not self.is_on:
            return 
        with open(self.score_filename, 'a') as writer:
            line = []
            for m, conn, score in scores:
                pair = str(m) + ',' + str(conn) + ','+ str(score) 
                line.append(pair)
            writer.write(' '.join(line) + '\n')


    def write_conns_mat(self, conns, ld):
        if not self.is_on:
            return 

        with open(self.filename, 'a') as writer:
            ticks = ['___'] + [str(a).rjust(5, '_') for a in range(len(conns))]
            writer.write('_'.join(ticks) + '\n')
            for i in range(len(conns)):
                text = [str(i).ljust(2, ' ') +  ' |']
                for j in range(len(conns)):
                    if j in conns[i] or i in conns[j]:
                        # link delay should be symmetric
                        assert(ld[i][j] == ld[j][i])
                        text.append("{:5d}".format(int(ld[i][j])))
                    else:
                        text.append("     ")
                line = ' '.join(text)
                writer.write(line + '\n')


        

    def write_mat(self, A, comment):
        if not self.is_on:
            return 
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
        if not self.is_on:
            return 
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
        if not self.is_on:
            return 
        with open(self.filename, 'a') as writer:
            out_txt = ["{:d}".format(int(a)) for a in outs]
            in_txt = ["{:d}".format(int(a)) for a in ins]
            line = '[O:' + ' '.join(out_txt) + ']. [I('+str(len(ins))+'):' + ' '.join(in_txt) + ' ]'
            if comment is not None:
                writer.write(str(comment) + ' ' + line + '\n' )
            else:
                writer.write(line + '\n')

    def write_ucb(self, region, arm, samples):
        if not self.is_on:
            return 
        with open(self.filename, 'a') as writer:
            line = 'r:'+str(region)+' a:'+str(arm)+' -> '+ str(samples)
            writer.write(line + '\n')

    def write_list(self, data, comment):
        if not self.is_on:
            return 
        with open(self.filename, 'a') as writer:
            text = ["{:4d}".format(int(a)) for a in data]
            line = ' '.join(text)
            if comment is not None:
                writer.write(str(comment) + line + '\n')
            else:
                writer.write(line + '\n')

    def write_str(self, txt):
        if not self.is_on:
            return 
        with open(self.filename, 'a') as writer:
            writer.write(txt + '\n')

    def format_mat(self, A, label_ticks, is_float):
        if not self.is_on:
            return 
        lines = []
        if label_ticks is not None:
            if not is_float:
                ticks = [str(a).rjust(4, '_') for a in label_ticks]
            else:
                ticks = [str(a).rjust(5, '_') for a in label_ticks]
        else:
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
        if not self.is_on:
            return 
        lines = []
        lines.append('_' + ''.join(title)+'_')
        for i in range(len(a)):
            text = ["{:5s}".format(str(a[i]))]
            line = ' '.join(text)
            lines.append(' ' + line + ' ')
        return lines

    def format_score(self, scores):
        if not self.is_on:
            return 
        lines = []
        title = '_Rslt'
        lines.append('_' + ''.join(title)+'_')
        for i in range(len(a)):
            text = ["{:5s}".format(str(a[i]))]
            line = ' '.join(text)
            lines.append(' ' + line + ' ')
        return lines


    def format_mat_5(self, A, is_float):
        if not self.is_on:
            return 
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
        if not self.is_on:
            return 
        lines = []
        num_mat = len(mats)
        num_line = len(mats[0])
        for i in range(num_line): 
            line = ''
            for mat in mats:
                line += mat[i] + '\t'
            print(line) 

    def log_mats(self, mats):
        if not self.is_on:
            return 
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
        if not self.is_on:
            return 
        for i in range(A.shape[0]):
            if not is_float:
                text = ["{:4d}".format(int(a)) for a in A[i]]
            else:
                text = ["{:5.2f}".format(a) for a in A[i]]
            line = ' '.join(text)
            print('[' + line + ' ]')

    def format_masked_mat(self, A, mask, label_ticks, is_float):
        if not self.is_on:
            return 
        lines = []
        if label_ticks is not None:
            if not is_float:
                ticks = [str(a).rjust(4, '_') for a in label_ticks]
            else:
                ticks = [str(a).rjust(5, '_') for a in label_ticks]
        else:
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

    def format_double_masked_mat(self, A, mask, none_mask, label_ticks, is_float):
        if not self.is_on:
            return 
        lines = []
        if label_ticks is not None:
            if not is_float:
                ticks = [str(a).rjust(4, '_') for a in label_ticks]
            else:
                ticks = [str(a).rjust(5, '_') for a in label_ticks]
        else:
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
                    elif none_mask[i,j] == 1:
                        text.append("{:>4}".format('+')) 
                    elif mask[i,j] == 0:
                        text.append("{:>4}".format('*')) 


            else:
                for j in range(len(row)):
                    assert(mask[i,j] == 0 and none_mask[i,j] == 1)
                    if mask[i,j] == 1:
                        text.append("{:5.2f}".format(row[j])) 
                    elif none_mask[i,j] == 1:
                        text.append("{:>5}".format('+')) 
                    elif mask[i,j] == 0:
                        text.append("{:>5}".format('*')) 
                    

            line = ' '.join(text)
            lines.append('[' + line + ' ]')
        return lines
