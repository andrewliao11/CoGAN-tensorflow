import numpy as np
from tqdm import tqdm 

f = open('list_attr_celeba.txt')
s = f.read()
s = s.split('\n')
s.pop(0)
s.pop(0)
s.pop(-1)
with_attr = []
without_attr = []
# default to be blonde hair (number 10)
attr = 10
for i in tqdm(s):
    data = i.split(' ')
    img = data[0]
    # filter out the ''
    data = filter(lambda item: item != '', data)
    if '\r' in data[attr]:
	data[attr].replace('\r', '')
    if data[attr] == '1':
	with_attr.append(img)
    else:
	without_attr.append(img)

np.savez('split_img', w_attr=with_attr, wo_attr=without_attr)

