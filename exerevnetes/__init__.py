import os

CWD = os.path.dirname(__file__)
VERSION_PATH = os.path.join(CWD, "_version.py")
__file__ = os.path.abspath(__file__)

with open(VERSION_PATH, "rt") as vf:
    __version__ = vf.read().split()[-1].strip('"')
    