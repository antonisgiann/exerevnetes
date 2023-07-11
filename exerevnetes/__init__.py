import os

#__file__ == os.path.abspath("..")
CWD = os.path.dirname(__file__)
VERSION_PATH = os.path.join(CWD, "_version.py")

with open(VERSION_PATH, "rt") as vf:
    __version__ = vf.read().split()[-1].strip('"')