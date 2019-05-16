import pyfftw
import numpy as np
import matplotlib.pyplot as plt

def taba(a1):
    '''Transform signal to Fourier domain and back.'''
    kwargs  = {'norm': None,}
    A       = pyfftw.interfaces.numpy_fft.fft(a1, **kwargs) # discrete transform
    a2      = pyfftw.interfaces.numpy_fft.ifft(A, **kwargs) # discrete siganl after ifft
    return a2

## Initialization
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# enable caching of FFTW objects. Also invokes the threading module.
# pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.disable()

func    = lambda x: np.exp(-x**2)

trange  = (-100.,100.)
dt      = .01
fs      = (2.*dt)**-1                   # Nyquist frequency
print('Nyquist freq.: {}'.format(fs))
Nt      = int((trange[1]-trange[0])/dt) + 1
t       = np.linspace(trange[0], trange[1], Nt)

a1      = pyfftw.byte_align(func(t), n=16, dtype='complex128')

a2      = a1.copy()
N       = 500
diff    = np.zeros(N)
print('pass 0, diff.: {}'.format(norm(a1-a2)))

## Calculation
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ti = time()
for i in range(N):
    a2      = taba(a2)
    diff[i] = norm(a1-a2)
print('pass {:d}, diff.: {:.3e}'.format(N, diff[N-1]))
delta_t = time()-ti
print("Transforming taba {} times took {:.3f} seconds.\nThat's {:.3e} seconds each on average".\
    format(N, delta_t, delta_t/N))

## Visualization
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.plot(t, np.abs(a), 'k')
plt.xlim([t[0],t[-1]])
plt.xlabel('t')
plt.ylabel('a')
plt.show()