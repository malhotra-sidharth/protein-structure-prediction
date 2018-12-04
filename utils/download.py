#!/usr/bin/env python
import os.path

f = open('yeastpdb.txt', "r")
lines = f.readlines()
f.close()

c=0

for line in lines:
	c += 1
	st=(line.strip()).lower()
	cmd="curl -O https://files.rcsb.org/download/"+st+".pdb"
	print(cmd)
	os.system(cmd)
#	print(cmd)
#	if (c > 5):
#		break
	print(c)