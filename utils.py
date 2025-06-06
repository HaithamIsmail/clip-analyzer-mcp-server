import os
import re

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    return [int(text) if text.isdigit() else text for text in parts]