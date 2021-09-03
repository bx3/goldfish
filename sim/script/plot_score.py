#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
if len(sys.argv) < 3:
    print('require input_file, outfile')
    sys.exit(1)

filename = sys.argv[1]
outname = sys.argv[2]
s_line = []
miners = []
with open(filename) as f:
    for line in f:
        if 'miners' in line:
            s = line.index(':')+1
            miners = line[s:].split()
        else:
            scores_str = line.split()
            data = [] 
            for score_str in scores_str:
                m, conn, score = score_str.split(',')
                data.append((int(m), int(conn), float(score)))

            s_line.append(data)

print(miners)
print(s_line)

miners_scores = defaultdict(list)
indices = []

for line in s_line:
    conns = []
    for slot in line:
        m, conn, score = slot 
        miners_scores[m].append(score)
        conns.append(conn)
    conns_str = [str(c) for c in conns]
    indices.append(' '.join(conns_str))



num_epoch = len(s_line)
epochs = [str(i) for i in range(num_epoch)]

df = pd.DataFrame(data=miners_scores)
df.index = indices #['C1', 'C2', 'C3']

plt.style.use('ggplot')

ax = df.plot(stacked=True, kind='bar', figsize=(12, 8), rot='horizontal')

# miner_labels = [str(i) for i in miners]
# for i, scores in miners_scores.items():
    # ax.bar(epochs, scores, label=str(i) )
plt.xticks(rotation=270)
ax.set_ylabel('')
# plt.xticks(epochs, indices, rotation=270)
ax.legend()
plt.savefig(outname)

