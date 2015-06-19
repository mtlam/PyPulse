PyPulse
=======

A python package for handling and analyzing PSRFITS files.

This is an alternate code base from [PSRCHIVE](http://psrchive.sourceforge.net/).

Requires:

numpy
scipy
matplotlib
astropy or pyfits


Archive
-------


Usage: 

    ar = Archive(FILENAME)
    ar.tscrunch() #averages the pulse in time