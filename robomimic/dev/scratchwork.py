import os
import nexusformat.nexus as nx
import h5py
import json

fn = "/home/nadun/Downloads/data.json"

with open(fn, 'rb')as f:
    data = json.load(f)

print()

