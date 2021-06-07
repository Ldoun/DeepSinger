import numpy as np

class duration_extractor(object):
    def __init__(self, alignment):
        super().__init__()

        self.alignment = alignment
        self.reward = np.zeros((self.alignment.size(0),self.alignment.size(1)))
        self.prefix_sum = self._get_prefix()
        self.splitting_boundary = np.zeros((self.alignment.size(0),self.alignment.size(1)))

    def _get_prefix(self):
        temp = np.zeros((self.alignment.size(0),self.alignment.size(1)))
        for i,row in enumerate(self.alignment):
            for j in range(1,self.alignment.size(1) + 1):
                temp[i][j-1] = sum(row[:j])

        return temp 
