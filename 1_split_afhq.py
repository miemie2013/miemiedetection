
import random

ori_train = '../afhq/train_ori.txt'
ori_val = '../afhq/val_ori.txt'

save_train = '../afhq/train.txt'
save_val = '../afhq/val.txt'


give_to_val = 1500

train_0 = []
train_1 = []
train_2 = []
val = []
with open(ori_train, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.endswith(' 0'):
            train_0.append(line)
        elif line.endswith(' 1'):
            train_1.append(line)
        elif line.endswith(' 2'):
            train_2.append(line)

random.shuffle(train_0)
random.shuffle(train_1)
random.shuffle(train_2)


with open(ori_val, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        val.append(line)


content_train = ''
content_val = ''
for record in val:
    content_val += record + '\n'


for i, record in enumerate(train_0):
    if i < give_to_val:
        content_val += record + '\n'
    else:
        content_train += record + '\n'


for i, record in enumerate(train_1):
    if i < give_to_val:
        content_val += record + '\n'
    else:
        content_train += record + '\n'


for i, record in enumerate(train_2):
    if i < give_to_val:
        content_val += record + '\n'
    else:
        content_train += record + '\n'



with open(save_train, 'w', encoding='utf-8') as f:
    f.write(content_train)
    f.close()

with open(save_val, 'w', encoding='utf-8') as f:
    f.write(content_val)
    f.close()

a = 1



