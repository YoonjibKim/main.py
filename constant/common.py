class Constant:
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise Exception('impossible to assign + ' + name + '.')
        self.__dict__[name] = value

    def __delattr__(self, name):
        if name in self.__dict__:
            raise Exception('impossible to delete ' + name + '.')

    def __init__(self):
        self.dataset_path = './dataset/acndata_sessions_office_1.json'
        self.result_dir_path = './result'


import sys

sys.modules[__name__] = Constant()
