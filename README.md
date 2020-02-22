PyPulse
=======

A pure-Python package for handling and analyzing PSRFITS files.

Read the documentation [here](https://mtlam.github.io/PyPulse/).

This is an alternate code base from [PSRCHIVE](http://psrchive.sourceforge.net/).

Requires:<br>
python 2.7 or 3.X<br>
numpy<br>
scipy<br>
matplotlib<br>
astropy or pyfits


Archive
-------

A class for loading PSRFITS files

Usage: 

    ar = Archive(FILENAME) #loads archive, dedispersed and polarization averaged by default
    ar.tscrunch() #averages the pulse in time
    data = ar.getData() #returns the numpy data array for use by you
    ar.imshow() #plots frequency vs phase for the pulses


SinglePulse
-----------

A class for handling calculations on single pulses

Usage:

    sp = SinglePulse(data,windowsize=256) #will auto-calculate an offpulse region of length 256 bins
    print sp.getFWHM() #prints the FWHM of the pulse
    print sp.getSN() #prints a crude S/N of the pulse
    print sp.fitPulse(template_array)[5] #prints a better S/N of the pulse using a template array

DynamicSpectrum
---------------

A class for handling dynamic spectra


Usage:

    ds = DynamicSpectrum(FILENAME) #takes a 2D data array or a string to a .npz file for now
    acf2d = ds.acf2d() #calculates the 2D ACF of the dynamic spectrum
    ds.imshow() #plots the dynamic spectrum
    ds.save(NEWFILENAME) #saves to a .npz file, which can then be loaded again with the first line

Par
---

A class for parsing pulsar parameter files

Usage:

    p = Par(FILENAME)
    print p.getPeriod() #prints the period in the par file (does not calculate for a specific MJD for now)
    t,dmx,dmxerr = p.getDMseries() #returns the DM parameters in a nice time-series. Note that errors are those given in the par file, which may not be the "correct" ones (more info to be provided later)
    print p.get("PX") #prints the parallax
    print p.get("PX",error=True) #prints the error on the parallax


Tim
---

A class for parsing pulsar TOA files

Usage:

    t = Tim(FILENAME) #automatically parses each TOA with the TOA() class
    print t.toas[0].getMJD() #prints the MJD of the first TOA as a python Decimal
    print t.toas[0].get('snr') #prints the value of the -snr flag of the TOA if it is available


Citations
---------

See [http://ascl.net/1706.011](http://ascl.net/1706.011). You can cite as:

[Lam, M. T., 2017, PyPulse, Astrophysics Source Code Library, record ascl:1706.011](http://adsabs.harvard.edu/abs/2017ascl.soft06011L)