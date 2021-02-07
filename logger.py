import sys


class Logger:
    def __init__(self, dirname, node_id):
        self.id = node_id
        self.filename = dirname + '/' + str(node_id)
        with open(self.filename, 'w') as writer:
            writer.write('node ' + str(node_id) + '\n')

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
