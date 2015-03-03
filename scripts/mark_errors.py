#!/usr/bin/python
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

import sys


for line in sys.stdin:
    t1, t2 = line.strip().split('\t')[-2:]
    if t1 == t2:
        print line.strip()
    else:
        print '{0}\t!'.format(line.strip())
