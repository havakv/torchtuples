import time
import random

def make_name_hash(name='', file_ending='.pt'):
    timestamp = time.ctime().replace(' ', '_')
    nanoseconds = str(time.time_ns())[-9:]
    ascii_letters_digits = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    random_hash = ''.join(random.choices(ascii_letters_digits, k=10))
    path = f"{name}_{timestamp}_{nanoseconds}_{random_hash}{file_ending}"
    return path