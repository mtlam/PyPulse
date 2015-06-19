'''
Michael Lam 2015

To do: grid: set clim such that 0 is white, not blue,mask data,add zorder
'''


import numpy as np
import numpy.ma as ma
import utils as u
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class DynamicSpectrum:
    def __init__(self,data,offdata=None,errdata=None,mask=None,F=None,T=None,extras=None,verbose=True):
        self.verbose=verbose
        if type(data) == type(""):
            name = data
            self.load(data)
            return
#            self.extras = dict()
#            self.extras['filename'] = data
        else:            
            self.data = data
        #check if 1d array
        if len(np.shape(self.data))>=1: #why==?????

            self.offdata = offdata
            self.errdata = errdata
            self.mask = mask
            self.F = F#these are the edges
            self.T = T
            self.Fcenter = None
            self.Tcenter = None
            if F is not None:
                self.Fcenter = np.diff(self.F)/2.0 + self.F[:-1]
            if T is not None:
                self.Tcenter = np.diff(self.T)/2.0 + self.T[:-1]
            if extras is None:
                self.extras = dict()
            else:
                self.extras = extras

    def getValue(self,f,t,df=1,dt=1,err=False,index=False):
        '''
        Returns value of dynamic spectrum
        if index==False, f,t are values to search for
        '''
        if index or self.F is None or self.T is None:
            if err:
                return self.errdata[f,t]
            return self.data[f,t]
        else:
            indsF = np.where(np.abs(self.Fcenter-f) <= df)[0]
            indsT = np.where(np.abs(self.Tcenter-t) <= dt)[0]
            if len(indsF)==0 or len(indsT)==0:
                return None
            if err:
                data = self.errdata
            else:
                data = self.data            
            total=0
            N=0
            for indF in indsF:
                for indT in indsT:
                    total+=data[indF,indT]
                    N+=1
            return total/float(N)
        
    def acf2d(self,speed='fast',mode='full'):
        """
        Calculate the two-dimensional auto-correlation function of the dynamic spectrum
        """
        return u.acf2d(self.data,speed=speed,mode=mode) #do more here


    def scintillation_parameters(self):
        
        pass






    def imshow(self,err=False,cbar=False,ax=None,show=True,border=False,ZORDER=0,cmap=cm.binary,alpha=True):
        """
        Basic plotting of the dynamic spectrum
        """
        if err:
            spec = self.errdata
        else:
            spec = self.data
        T=self.T
        if self.T is None:
            T=self.Tcenter
        F=self.F
        if self.F is None:
            F=self.Fcenter
        #cmap = cm.binary#jet
        if alpha:
            cmap.set_bad(alpha=0.0)


        if alpha: #do this?
            for i in range(len(spec)):
                for j in range(len(spec[0])):
                    if spec[i][j]<=0.0:# or self.errdata[i][j]>3*sigma:
                        spec[i][j]=np.nan
        
        minT = T[0]
        maxT = T[-1]
        minF = F[0]
        maxF = F[-1]


#        print inds
#        raise SystemExit
#        spec[inds] = np.nan
        im=u.imshow(spec,ax=ax,extent = [minT,maxT,minF,maxF],cmap=cmap,zorder=ZORDER)

        #border here?
        if border:# and self.extras['name']!='EFF I':
            plt.plot([T[0],T[-1]],[F[0],F[0]],'0.50',zorder=ZORDER+0.1)
            plt.plot([T[0],T[-1]],[F[-1],F[-1]],'0.50',zorder=ZORDER+0.1)
            plt.plot([T[0],T[0]],[F[0],F[-1]],'0.50',zorder=ZORDER+0.1)
            plt.plot([T[-1],T[-1]],[F[0],F[-1]],'0.50',zorder=ZORDER+0.1)


        if cbar:
            plt.colorbar()
        #im.set_clim(0.0001,None)
        if show:
            plt.show()

        return ax


    def load(self,filename):
        """
        Load the dynamic spectrum from a .npz file
        """
        if self.verbose:
            print "Dynamic Spectrum: Loading from file: %s" % filename
        x = np.load(filename)
        for key in x.keys():
            exec("self.%s=x['%s']"%(key,key))
        exec("self.extras = dict(%s)"%self.extras)
        #Convert array(None) to None
        if self.offdata is not None and len(np.shape(self.offdata))==0:
            self.offdata = None
        if self.errdata is not None and len(np.shape(self.errdata))==0:
            self.errdata = None
        if self.mask is not None and len(np.shape(self.mask))==0:
            self.mask = None
            
        

        x.close()
        return

    def save(self,filename):
        """
        Save the dynamic spectrum to a .npz file
        """
        if self.verbose:
            print "Dynamic Spectrum: Saving to file: %s" % filename
        np.savez(filename,data=self.data,offdata=self.offdata,errdata=self.errdata,mask=self.mask,F=self.F,T=self.T,Fcenter=self.Fcenter,Tcenter=self.Tcenter,extras=self.extras)
        return

        
    # Must be in time order!
    def add(self,ds,axis='T'):
        """
        Concatenates another dynamic spectrum with this one
        """
        if axis=='T':
            self.T = np.concatenate((self.T,ds.T))
            if len(np.shape(ds.data))==1:
                ds.data = np.reshape(ds.data,[len(ds.data),1])
                if ds.offdata is not None:
                    ds.offdata = np.reshape(ds.offdata,[len(ds.offdata),1])
                if ds.errdata is not None:
                    ds.errdata = np.reshape(ds.errdata,[len(ds.errdata),1])
                if ds.mask is not None:
                    ds.mask = np.reshape(ds.mask,[len(ds.mask),1])


            self.data = np.hstack((self.data,ds.data))
            #if statements
            if self.offdata is None and ds.offdata is None:
                self.offdata = None
            else:
                if self.offdata is None and ds.offdata is not None:
                    self.offdata = np.zeros(np.shape(self.data))
                elif self.offdata is not None and ds.offdata is None:
                    ds.offdata = np.zeros(np.shape(ds.data))
                self.offdata = np.hstack((self.offdata,ds.offdata))
            if self.errdata is None and ds.errdata is None:
                self.errdata = None
            else:
                if self.errdata is None and ds.errdata is not None:
                    self.errdata = np.zeros(np.shape(self.data))
                elif self.errdata is not None and ds.errdata is None:
                    ds.errdata = np.zeros(np.shape(ds.data))
                self.errdata = np.hstack((self.errdata,ds.errdata))
            if self.mask is None and ds.mask is None:
                self.mask = None
            else:
                if self.mask is None and ds.mask is not None:
                    self.mask = np.zeros(np.shape(self.data))
                elif self.mask is not None and ds.mask is None:
                    ds.mask = np.zeros(np.shape(ds.data))
                self.mask = np.hstack((self.mask,ds.mask))
            
            #Regenerate Tcenter?
            #Add extras together?




