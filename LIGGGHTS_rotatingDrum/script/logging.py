#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 08:42:25 2021

@author: tobias
"""

import logging
import sys

import colorlog


log = logging.getLogger('log1')
for h in log.handlers:
    log.removeHandler(h)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)1.1s %(asctime)s %(module)s]%(reset)s %(message)s"
    ))
log.addHandler(handler)
log.setLevel(logging.DEBUG)
log.propagate = False

#log.warning('warn2')
#log.info('info')
#log.debug('info')

def getLogger():
    return logging.getLogger('log1')