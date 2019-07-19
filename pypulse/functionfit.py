import numpy as np
import scipy.optimize as optimize
import scipy.special as special


def funcgaussian(p, x, baseline=False):
    if baseline:
        return p[0] * np.exp(-((x-p[1])/(np.sqrt(2)*p[2]))**2) + p[3]
    return p[0] * np.exp(-((x-p[1])/(np.sqrt(2)*p[2]))**2)

def errgaussian(p, x, y, baseline=False):
    return funcgaussian(p, x, baseline=baseline) - y

def gaussianfit(x, y, baseline=False):
    x = np.array(x)
    y = np.array(y)
    height = max(y)
    mu = np.sum(x*y)/np.sum(y)
    sigma = np.sqrt(np.abs(np.sum((x-mu)**2*y)/np.sum(y)))
    if baseline:
        p0 = [height, mu, sigma, 0.0]
    else:
        p0 = [height, mu, sigma]
    p1, success = optimize.leastsq(errgaussian, p0[:], args=(x, y, baseline))
    p1[2] = np.abs(p1[2]) #enforces positive sigma
    #Return values are the coefficients, the residuals
    return p1, errgaussian(p1, x, y, baseline)

#area = np.sum(binwidths*hist)
def funcsimpleDISSpdf(p, x, area=None):
    if area is None:
        area = 1
    S0 = p[0]
    niss = p[1]
    #scale = p[2]
    g = x/S0
    return area*((g*niss)**niss / (g * special.gamma(niss))) * np.exp(-g*niss) /S0
    #return area*((g*niss)**niss / (g * special.gamma(niss))) * np.exp(-g*niss) /S0
    #return scale*((g*niss)**niss / (g * special.gamma(niss))) * np.exp(-g*niss) /S0

def errsimpleDISSpdf(p, x, y, area=None):
    return funcsimpleDISSpdf(p, x, area=area) - y

def simpleDISSpdffit(x, y):
    x = np.array(x)
    y = np.array(y)
    p0 = [x[np.argmax(y)], 1.0]#, np.max(y)]
    p1, success = optimize.leastsq(errsimpleDISSpdf, p0[:], args=(x, y))
    #Return values are the coefficients, the residuals
    return p1, errsimpleDISSpdf(p1, x, y)

#2d Gaussian fitting
#Modification of scipy cookbook to include rotation http://wiki.scipy.org/Cookbook/FittingData
#Note! arr[r, c] convention versus (x, y)
def gaussian2d(amplitude, center_x, center_y, width_x, width_y, rotation, baseline):
    center_x_rot = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    center_y_rot = center_x * np.sin(rotation) + center_y * np.cos(rotation)
    def rotgauss(x, y):
        xrot = x * np.cos(rotation) - y * np.sin(rotation)
        yrot = x * np.sin(rotation) + y * np.cos(rotation)
        return amplitude*np.exp(-0.5*(((center_x_rot-xrot)/width_x)**2+((center_y_rot-yrot)/width_y)**2)) + baseline #Baseline leads to underestimated scintillation parameters and is unnecessary if a baseline value is removed prior?
    return rotgauss

def moments(data):
    """Returns (height, x, y, width_x, width_y, rotation, baseline)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = np.sum(data)
    X, Y = np.indices(data.shape)
    x = np.sum(X*data)/total
    y = np.sum(Y*data)/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = np.max(data)
    baseline = np.min(data)
    return height-baseline, x, y, width_x, width_y, 0.0, baseline

def fitgaussian2d(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian2d(*p)(*np.indices(data.shape)) - data)
    p, cov, infodict, mesg, ier = optimize.leastsq(errorfunction, params, full_output=True)
    #Must multiply by reduced chi-squared first
    #http://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i
    s_sq = (errorfunction(p)**2).sum()/(data.size - len(p))
    if cov is not None:
        cov *= s_sq
    return p, cov


