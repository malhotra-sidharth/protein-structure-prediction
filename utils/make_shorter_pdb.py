#!/usr/bin/python

import os, sys
import re

# Open a file
dirs = os.listdir(".")
if not os.path.exists("shortpdb"):
	os.mkdir("shortpdb")


c =0
# This would print all the files and directories
for file in dirs:
	c +=1
	if(re.search('pdb$', file)):
		print file
		cmd = "grep HELIX "+file+" > shortpdb/"+file
		print(cmd)
		os.system(cmd)
		os.system("grep SEQRES "+file+" >> shortpdb/"+file)
		os.system("grep DBREF "+file+" >> shortpdb/"+file)
	#if(c > 5):
	#	break