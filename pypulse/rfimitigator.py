

import numpy as np



class RFImitigator:
    def __init__(self,archive):
        self.archive = archive


    def can_mitigate(self,mode='frequency'):
        '''
        Check if the shape of the data array is conducive to mitigation
        '''
        if mode == 'frequency':
            INDEX = 2

        SHAPE = self.archive.shape(squeeze=False)
        if SHAPE[2] == 1:
            return False
        else:
            return True
        
    def zap_frequency_range(self,nulow,nuhigh):
        '''
        Mitigate channels within a given frequency range
        '''
        if not self.can_mitigate():
            return

        Faxis = self.archive.getAxis('F')
        inds = np.where((nulow<=Faxis)&(Faxis<=nuhigh))[0]
        for ind in inds:
            self.archive.setWeights(0.0,f=ind)

        
    def zap_channels(self,index):
        '''
        Mitigate an individual channel
        '''
        if not self.can_mitigate():
            return
        if isinstance(index,int):
            self.archive.setWeights(0.0,f=index)
        elif isinstance(index,(list,tuple,np.ndarray)):
            for elem in index:
                self.archive.setWeights(0.0,f=elem)

        
    def zap_minmax(self):
        '''
        Run NANOGrav algorithm
        '''
        if not self.can_mitigate():
            return

        SHAPE = self.archive.shape(squeeze=False)
