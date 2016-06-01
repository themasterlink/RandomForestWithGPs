#!/usr/bin/python

import urllib

urllib.urlretrieve ("http://www.image-net.org/api/text/imagenet.sbow.obtain_synset_list", "data/list.txt")


lines = [line.rstrip('\n') for line in open('data/list.txt')]

for line in lines:
    if len(line) > 0:
        urllib.urlretrieve ("http://www.image-net.org/downloads/features/vldsift/" + line + ".vldsift.mat", "data/" + line + ".mat")
