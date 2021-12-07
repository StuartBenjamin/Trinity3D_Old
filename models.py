import numpy as np

### this library contains model functons for flux behavior

def ReLU(x,a=0.5,m=1):
    '''
       piecewise-linear function
       can model Gamma( critical temperature gradient scale length ), for example
       x is a number, a and m are constants

       inputs : a number
       outputs: a number
    '''
    if (x < a):
        return 0
    else:
        return m*(x-a)

def Step(x,a=0.5,m=1):
    # derivative of ReLU (is just step function)
    if (x < a):
        return 0
    else:
        return m


# for a particle source
def Gaussian(x,sigma=.3,A=2):
    w = - (x/sigma)**2  / 2
    return A * np.e ** w
