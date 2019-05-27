import numpy as np
import matplotlib.pyplot as plt
t=np.arange(0, 10, 1)

fv= lambda t: np.sqrt(9.81*68/0.25)*np.tanh(np.sqrt(9.81*0.25/68)*t)

fx= lambda t : 68/0.25*np.log(np.cosh(np.sqrt(9.8*0.25/68)*t))

fv_t=fv(t)
fx_t=fx(t)
plt.figure(104)
vel, =plt.plot(t, fv_t, 'b*-', label='Velocity[m/s]')
dis, =plt.plot(t, fx_t, 'ro-', label='Distance[m]')
plt.legend(handles=[vel, dis])
plt.grid()
plt.xlabel('Sec')
plt.ylabel('Velocity[m/s], Distance[m]')
