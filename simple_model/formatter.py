import sys
from collections import defaultdict

def get_topo_direction(slots, log_directions, num_topo):
    topo_directions = {}
    id_set = set()
    num_slot = int(len(slots) / num_topo)
    for i, slot in enumerate(slots):
        topo_i = int(i / num_slot)
        if topo_i not in topo_directions:
            topo_directions[topo_i] = defaultdict(set)

        for p, t, direction in slot:
            if direction in log_directions:
                id_set.add(p)
                topo_directions[topo_i][p].add(direction)

    # for i in range(num_topo):
        # topo_directions[i] = sorted((topo_directions[i]).items())

    # N = len(id_set)
    # T = len(slots)
    # id_map = {}
    
    # for i in range(N):
        # id_map[ids[i]] = i

    
    # i = 0 # row
    # X_topo = {}
    # mask_topo = {}
    # none_mask_topo = {}
    # max_time = 0 
    # for slot in slots:
        # topo_i = int(i / num_slot)
        # if topo_i not in X_topo:
            # X_topo[topo_i] = np.zeros((T, N)) 
            # mask_topo[topo_i] = np.zeros((T, N))
            # none_mask_topo[topo_i] = np.zeros((T, N))

        # for p, t, direction in slot:
            # if direction in log_directions:
                # if t is not None:
                    # j = id_map[p]
                    # X_topo[topo_i][i, j] = t
                    # mask_topo[topo_i][i, j] = 1
                    # max_time = max(t, max_time)
                # else:
                    # j = id_map[p]
                    # none_mask_topo[topo_i][i,j] = 1
        # i += 1
    return topo_directions

def print_curr_conns(conns, ld):            
    ticks = ['____'] + [str(a).rjust(5, '_') for a in range(len(conns))]
    print('_'.join(ticks))
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
        print(line )

def format_direction_ticks(topo_direction, s, has_direct, is_float):
    num_row, num_col = s
    num_topo = len(topo_direction)
    ticks_list = []
    all_conns = set()
    for i, label_ticks in topo_direction.items():
        all_conns = all_conns.union(label_ticks.keys())
    all_conns = sorted(list(all_conns)) 
    for i in range(num_topo):
        label_ticks = topo_direction[i]
        ticks = []
        num_spacer = 4
        if is_float:
            num_spacer = 5
        for peer in all_conns:
            if peer not in label_ticks:
                ticks.append(('x'+str(peer)).rjust(num_spacer, '_'))
            else:
                directions = label_ticks[peer]
                assert(len(directions) > 0 and len(directions) < 3)
                if 'bidirect' in directions:
                    ticks.append(('*'+str(peer)).rjust(num_spacer, '_'))
                elif 'incoming' in directions:
                    ticks.append(('-'+str(peer)).rjust(num_spacer, '_'))
                elif 'outgoing' in directions :
                    ticks.append(('+'+str(peer)).rjust(num_spacer, '_'))
                else:
                    print('Error. Unknown direction')
                    sys.exit(1)
        ticks_list.append(ticks)
    return ticks_list

def format_ticks(label_ticks, s, has_direct, is_float):
    num_row, num_col = s
    ticks_list = []
    if label_ticks is not None:
        if not has_direct:
            if not is_float:
                ticks = [str(a).rjust(4, '_') for a in label_ticks]
            else:
                ticks = [str(a).rjust(5, '_') for a in label_ticks]
            ticks_list.append(ticks)
        else:
            ticks = []
            num_spacer = 4
            if is_float:
                num_spacer = 5

            for peer, directions in label_ticks:
                assert(len(directions) > 0 and len(directions) < 3)
                # print(peer)
                # print(directions)
                # sys.exit(1)
                if 'bidirect' in directions:
                    ticks.append(('*'+str(peer)).rjust(num_spacer, '_'))
                elif 'incoming' in directions:
                    ticks.append(('-'+str(peer)).rjust(num_spacer, '_'))
                elif 'outgoing' in directions :
                    ticks.append(('+'+str(peer)).rjust(num_spacer, '_'))
                else:
                    print('Error. Unknown direction')
                    sys.exit(1)
    else:
        if not is_float:
            ticks = [str(a).rjust(4, '_') for a in range(num_col)]
        else:
            ticks = [str(a).rjust(5, '_') for a in range(num_col)]
    return ticks

def format_mat(A, topo_directions, num_topo, has_direct, is_float):
    lines = []
    ticks_list = format_direction_ticks(topo_directions, A.shape, has_direct, is_float)
    assert(A.shape[0] % num_topo == 0)
    block_size = int(A.shape[0] / num_topo)
    tick_i = 0
    for i in range(A.shape[0]):
        if not is_float:
            text = ["{:4d}".format(int(a)) for a in A[i]]
        else:
            text = ["{:5.2f}".format(a) for a in A[i]]
        if i % block_size == 0:
            lines.append('_' + '_'.join(ticks_list[tick_i]) + '__')
            tick_i += 1

        line = ' '.join(text)
        lines.append('[' + line + ' ]')
    return lines

def format_array(a, title):
    lines = []
    lines.append('_' + ''.join(title)+'_')
    for i in range(len(a)):
        text = ["{:5s}".format(str(a[i]))]
        line = ' '.join(text)
        lines.append(' ' + line + ' ')
    return lines

def format_topo_array(a, num_topo, title):
    lines = []
    lines.append('_' + ''.join(title)+'_')
    assert(len(a) % num_topo == 0)
    block_size = int(len(a) / num_topo)

    for i in range(len(a)):
        if i!=0 and i%block_size==0:
            lines.append('-------')
        text = ["{:5s}".format(str(int(a[i])))]
        line = ' '.join(text)
        lines.append(' ' + line + ' ')
    return lines



def format_double_masked_mat(A, mask, none_mask, topo_directions, num_topo, has_direct, is_float):
    lines = []
    ticks_list = format_direction_ticks(topo_directions, A.shape, has_direct, is_float)
    assert(A.shape[0] % num_topo == 0)
    block_size = int(A.shape[0] / num_topo)
    tick_i = 0

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
                if mask[i,j] == 1:
                    text.append("{:5.2f}".format(row[j])) 
                elif none_mask[i,j] == 1:
                    text.append("{:>5}".format('+')) 
                elif mask[i,j] == 0:
                    text.append("{:>5}".format('*')) 
        if i % block_size == 0:
            lines.append('_' + '_'.join(ticks_list[tick_i]) + '__')
            tick_i += 1           

        line = ' '.join(text)
        lines.append('[' + line + ' ]')
    return lines

def format_completed_mat(A, unkn_unab_mask, unkn_plus_mask, kn_plus_mask, topo_directions, num_topo, has_direct, is_float):
    lines = []
    ticks_list = format_direction_ticks(topo_directions, A.shape, has_direct, is_float)
    assert(A.shape[0] % num_topo == 0)
    block_size = int(A.shape[0] / num_topo)
    tick_i = 0

    for i in range(A.shape[0]):
        row = A[i]
        text = []
        if not is_float:
            for j in range(len(row)):
                if kn_plus_mask[i,j] == 1:
                    text.append("{:>4}".format('+')) 
                elif unkn_unab_mask[i,j] == 1:
                    text.append("{:>4}".format('x')) 
                elif unkn_plus_mask[i,j] == 1:
                    text.append("{:>4}".format('&')) 
                elif unkn_unab_mask[i,j] == 0:
                    text.append("{:4d}".format(int(row[j]))) 
        else:
            for j in range(len(row)):
                if kn_plus_mask[i,j] == 1:
                    text.append("{:>5}".format('+')) 
                elif unkn_unab_mask[i,j] == 1:
                    text.append("{:>5}".format('x')) 
                elif unkn_plus_mask[i,j] == 1:
                    text.append("{:>5}".format('&')) 
                elif unkn_unab_mask[i,j] == 0:
                    text.append("{:5.2f}".format(row[j])) 


        if i % block_size == 0:
            lines.append('_' + '_'.join(ticks_list[tick_i]) + '__')
            tick_i += 1           

        line = ' '.join(text)
        lines.append('[' + line + ' ]')
    return lines

def print_mats(mats, num_topo):
    lines = []
    num_mat = len(mats)
    num_line = len(mats[0])
    assert(num_line % num_topo == 0)
    block_size = int(num_line / num_topo)
    tick_i = 0

    for i in range(num_line): 
        if i % block_size == 0:
            tick_i += 1
            line = "  |"
        else:
            line = "{:<2}|".format(str(i - tick_i))

        for mat in mats:
            line += mat[i] + '\t'
        print(line) 
