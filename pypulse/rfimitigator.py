import sys
import numpy as np
if sys.version_info.major == 2:
    fmap = map
elif sys.version_info.major == 3:
    fmap = lambda x, *args: list(map(x, *args))
    xrange = range

class RFIMitigator(object):
    def __init__(self, archive):
        self.archive = archive

    def can_mitigate(self, flag='F'):
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

    def zap(self, val=0.0, t=None, f=None):
        '''
        Passes straight to archive's setWeights()
        '''
        self.archive.setWeights(val=val, t=t, f=f)

    def unzap(self):
        '''
        Gets rid of all weighting
        '''
        MAX = np.max(self.archive.getWeights())
        self.archive.setWeights(MAX)

    def zap_frequency_range(self, nulow, nuhigh):
        '''
        Mitigate channels within a given frequency range
        '''
        if not self.can_mitigate():
            return

        Faxis = self.archive.getAxis('F')
        inds = np.where((nulow <= Faxis)&(Faxis <= nuhigh))[0]
        for ind in inds:
            self.zap(f=ind)

    def zap_channels(self, index):
        '''
        Mitigate an individual channel
        '''
        if not self.can_mitigate():
            return
        if isinstance(index, int):
            self.archive.setWeights(0.0, f=index)
        elif isinstance(index, (list, tuple, np.ndarray)):
            for elem in index:
                self.zap(f=elem)

    def zap_minmax(self, windowsize=20, threshold=4):
        '''
        Run NANOGrav algorithm, median zapping. Run per subintegration
        windowsize = 20 frequency bins long
        threshold = 4 sigma
        '''
        if not self.can_mitigate():
            return

        nsubint = self.archive.getNsubint()
        nchan = self.archive.getNchan()

        # Prepare data
        data = self.archive.getData(squeeze=False)
        #SinglePulse average profile, no need to invoke creating more SinglePulse instances
        spavg = self.archive.spavg
        opw = spavg.opw

        if nchan <= windowsize:
            for i in xrange(nsubint):
                for j in xrange(nchan):
                    subdata = data[i, 0, :, opw]
                    compptp = np.ptp(data[i, 0, j, opw])
                    ptps = np.zeros(windowsize)
                    for k in xrange(windowsize):
                        ptps[k] = np.ptp(subdata[k, :])

                    med = np.median(ptps)
                    if compptp > threshold*med:
                        self.zap(f=j)
            return

        for i in xrange(nsubint):
            for j in xrange(nchan):
                low = j - windowsize//2
                high = j + windowsize//2

                if low < 0:
                    high = abs(low)
                    low = 0
                elif high > nchan:
                    diff = high - nchan
                    high -= diff
                    low -= diff

                subdata = data[i, 0, low:high, opw]
                compptp = np.ptp(data[i, 0, j, opw])
                ptps = np.zeros(windowsize)
                for k in xrange(windowsize):
                    ptps[k] = np.ptp(subdata[k, :])

                #ptps = np.array(map(lambda subdata: np.ptp(subdata),data[i,0,low:high,opw]))

                med = np.median(ptps)
                if compptp > threshold*med:
                    self.zap(f=j)

        return
