from os import mkdir
from os.path import exists


def create_folder(path = 'origin-data'):
    if not exists(path): mkdir(path)
    if not exists(path + 'fake-images'): mkdir(path + 'fake-images')
    if not exists(path + 'real-images'): mkdir(path + 'real-images')
    if not exists(path + 'test_set'): mkdir(path + 'test_set')
    for _ in range(25, 1040):
        tmp = f'fake-images/part-{_:06}'
        if not exists(tmp): mkdir(tmp)
        if not exists(tmp): mkdir(tmp)
    print('=> Create Complete')