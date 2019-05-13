import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy.fft import *

def taba(a1):
    '''Transform signal to Fourier domain and back.'''
    kwargs  = {'norm': None,}
    A       = fft(a1, **kwargs)          # discrete transform
    a2      = ifft(A, **kwargs)         # discrete siganl after ifft
    return a2

func    = lambda x: np.exp(-x**2)

trange  = (-100.,100.)
dt      = .01
fs      = (2.*dt)**-1                   # Nyquist frequency
print('Nyquist freq.: {}'.format(fs))
Nt      = int((trange[1]-trange[0])/dt) + 1
t       = np.linspace(trange[0], trange[1], Nt)

a1      = func(t)                       # discrete siganl before fft
freq    = fftfreq(Nt, d=dt)

a2      = a1
N       = 100000
diff    = np.zeros(N)
print('pass 0, diff.: {}'.format(norm(a1-a2)))
for i in range(N):
    a2      = taba(a2)
    diff[i] = norm(a1-a2)
    # print('pass {:d}, diff.: {:.3e}'.format(i, diff[i]))

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
plt.plot(t, a2, 'k')
plt.xlim(trange)
plt.title('signal')
plt.xlabel('$t$')
plt.ylabel('a')
plt.grid()

# plt.figure(3)
# plt.plot(fftshift(f), fftshift(np.abs(A)), 'k--')
# plt.xlim([-fs,fs])
# plt.title('transform')
# plt.xlabel('$\omega$')
# plt.ylabel('A')
# plt.grid()

plt.show()