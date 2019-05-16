import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy.fft import *
from timeit import timeit

def taba(a1):
    '''Transform signal to Fourier domain and back.'''
    kwargs  = {'norm': None,}
    A       = fft(a1, **kwargs)          # discrete transform
    a2      = ifft(A, **kwargs)         # discrete siganl after ifft
    return a2

def transf(a, N):
    '''Take a, make a copy and transform it N times taba.
    Return the resulting singal and the difference after each
    transformation pair.'''

    a2      = a.copy()
    diff    = np.zeros(N)
    for i in range(N):
        a2      = taba(a2)
        diff[i] = norm(a-a2)

    return a2, diff

## Initialization
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

func    = lambda x: np.exp(-x**2)

trange  = (-100.,100.)
dt      = .01
fs      = (2.*dt)**-1                   # Nyquist frequency
print('Nyquist freq.: {}'.format(fs))
Nt      = int((trange[1]-trange[0])/dt) + 1
t       = np.linspace(trange[0], trange[1], Nt)

a1      = func(t)                       # discrete siganl before fft
freq    = fftfreq(Nt, d=dt)

N       = 10000

## Calculation
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

delta_t = timeit('a2, diff = transf({:d})'.format(N), number=100,
    setup="from __main__ import transf")
print('pass {:d}, diff.: {:.3e}'.format(N, diff[N-1]))
print("Transforming taba {} times took {:.3f} seconds.\nThat's {:.3e} seconds each on average".\
    format(N, delta_t, delta_t/N))

## Visualization
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# plot normed difference vs. nr of transforms
plt.figure(1)
plt.plot(diff, 'k')
plt.xlim([0,N])
plt.title('Normed difference, N={}'.format(N))
plt.xlabel('number of transforms')
plt.ylabel('difference')
plt.grid()

# plot original and N times back-and-forth transformed signal
plt.figure(2)
plt.plot(t, a1, 'k--')
plt.plot(t, np.abs(a2), 'k')
plt.xlim(trange)
plt.title('signal')
plt.xlabel('$t$')
plt.ylabel('a')
plt.grid()

plt.show()