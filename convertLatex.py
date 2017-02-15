#! /usr/bin/env python
from os import listdir, remove, rename
from os.path import isfile, join, dirname, realpath

mypath = dirname(realpath(__file__))
#mypath = '/Users/SkinTang/Projects/markdownToBlog/'
file = [f for f in listdir(mypath) if isfile(join(mypath, f)) & (f[-2:] == 'md')]
for singlefile in file:
    f = open(join(mypath, singlefile),'r')
    filenamenew = singlefile.replace('.md','R.md')
    f2 = open(join(mypath, filenamenew),'w')
    for lines in f:
        lines2 = lines.replace('\\\\','\\\\\\\\')
        lines2 = lines2.replace('\(','$$')
        lines2 = lines2.replace('\)','$$')
        lines2 = lines2.replace('\[','\\\\[')
        lines2 = lines2.replace('\]','\\\\]')
        lines2 = lines2.replace('\{','\\\\{')
        lines2 = lines2.replace('\}','\\\\}')
        f2.writelines(lines2)
    f2.close()
    remove(singlefile)
    rename(filenamenew, singlefile)

