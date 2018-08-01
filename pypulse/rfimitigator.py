

import numpy as np



class RFIMitigator:
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


    def zap(self,val=0.0,t=None,f=None):
        '''
        Passes straight to archive's setWeights()
        '''
        self.archive.setWeights(val,t=t,f=f)
    def unzap(self):
        '''
        Gets rid of all weighting
        '''
        self.archive.setWeights(1.0)

        
    def zap_frequency_range(self,nulow,nuhigh):
        '''
        Mitigate channels within a given frequency range
        '''
        if not self.can_mitigate():
            return

        Faxis = self.archive.getAxis('F')
        inds = np.where((nulow<=Faxis)&(Faxis<=nuhigh))[0]
        for ind in inds:
            self.zap(f=ind)

        
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
                self.zap(f=elem)




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

        # Prepare data
        data = self.archive.getData(squeeze=False)
        spavg = self.archive.spavg #SinglePulse average profile, no need to invoke creating more SinglePulse instances
        opw = spavg.opw
        
        if nchan <= windowsize:
            for i,t in enumerate(nsubint):
                ptps = np.array(map(lambda subdata: np.ptp(subdata,axis=1),data[i,0,:,opw]))
                med = np.median(ptps)
                for j,f in enumerate(nchan):
                    if ptps[j] > threshold*med:
                        self.zap(f=ind)
            return

        
        for i,t in enumerate(nsubint):
            for j,f in enumerate(nchan):
                low = f - windowsize//2
                high = f + windowsize//2

                if low < 0:
                    high = abs(low)
                    low = 0
                elif high > nchan:
                    diff = high - nchan
                    high -= diff
                    low -= diff
                
                ptps = np.array(map(lambda subdata: np.ptp(subdata,axis=1),data[i,0,low:high,opw]))
                med = np.median(ptps)
                for j,f in enumerate(nchan):
                    if ptps[j] > threshold*med:
                        self.zap(f=ind)
                
        return
