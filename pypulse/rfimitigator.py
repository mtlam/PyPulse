

import numpy as np



class RFImitigator:
    def __init__(self,archive):
        self.archive = archive


    def can_mitigate(self,flag='F'):
        '''
        Check if the shape of the data array is conducive to mitigation
        '''
        if flag == 'F':
            if self.archive.getNchan() == 1:
                return False
            return True
        if flag == 'T':
            if self.archive.getNsubint() == 1:
                return False
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

        
    def zap_minmax(self,windowsize=20,threshold=4):
        '''
        Run NANOGrav algorithm, median zapping. Run per subintegration
        windowsize = 20 frequency bins long
        threshold = 4 sigma
        '''
        if not self.can_mitigate():
            return


        nsubint = self.getNsubint()
        nchan = self.getNchan()

        for i,t in enumerate(nsubint):
            for j,f in enumerate(nchan):
                return
