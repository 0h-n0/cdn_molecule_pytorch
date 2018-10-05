
class SmilesDataLoader(object):
    def __init__(self, filename):
        self.raw_data = None
        with open(filename, 'r') as fp:
            self.raw_data = fp.readlines()
        print(self.raw_data)

    def __getitem__(self, i):
        pass

if __name__ == '__main__':
    SmilesDataLoader
    




