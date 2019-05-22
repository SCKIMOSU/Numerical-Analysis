import numpy as np
import matplotlib.pyplot as plt

fs = 1000      # sampling frequency 1000 Hz
dt = 1/fs      # sampling period
N  = 1500      # length of signal

t  = np.arange(0,N)*dt   # time = [0, dt, ..., (N-1)*dt]
s = 0.7*np.sin(2*np.pi*60*t) + np.sin(2*np.pi*120*t)
x = s+2*np.random.randn(N)   # random number Normal distn, N(0,2)... N(0,2*2)

plt.subplot(2,1,1)
plt.plot(t[0:51],s[0:51],label='s')
plt.plot(t[0:51],x[0:51],label='x')
plt.legend()
plt.xlabel('time'); plt.ylabel('x(t)'); plt.grid()

# Fourier spectrum
df = fs/N   # df = 1/N = fmax/N
f = np.arange(0,N)*df     #   frq = [0, df, ..., (N-1)*df]
xf = np.fft.fft(x)*dt
plt.subplot(2,1,2)
plt.plot(f[0:int(N/2+1)],np.abs(xf[0:int(N/2+1)]))
plt.xlabel('frequency(Hz)'); plt.ylabel('abs(xf)'); plt.grid()
plt.tight_layout()
