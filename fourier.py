import numpy as np
import matplotlib.pyplot as plt

Fs = 150.0  # sampling rate
Ts = 1.0/Fs # sampling interval or sampling time
# 0.006666666666666667

t = np.arange(0,1,Ts) # time vector
# 0에서 1초 사이에 0.006 초 간격의 (Ts개) 시간을 만들어라
# np.size(t)=150

ff = 5   # frequency of the signal
y = np.sin(2*np.pi*ff*t)
# sin(2*np.pi)를 1 초 (t) 안에 5개 (ff)의 2*pi 주기의 sin 파를 만들어라. 
# 그런데, 1초 안에는 150개의 점을 찍어라.
# np.size(y)=150

#  compare sine and cosine
y1= np.cos(2*np.pi*ff*t)

########### draw sin and cosine
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,y, 'b.')
# np.size(t) = 150
# np.size(y) = 150
plt.xlabel('Time')
plt.ylabel('sin(t)')
plt.grid()
plt.subplot(2,1,2)
plt.plot(t,y1,'r*')
plt.xlabel('Time')
plt.ylabel('cos(t)')
plt.grid()

#### Reduce Sampling Rate

fs = 20.0  # sampling rate
ts = 1.0/fs # sampling interval or sampling time
# 0.05  <-- 0.006666666666666667

t1 = np.arange(0,1,ts) # time vector
# 0에서 1초 사이에 0.05 초 간격의 (ts개) 시간을 만들어라
# np.size(t1)=20

ff1 = 5   # frequency of the signal
yy = np.sin(2*np.pi*ff1*t1)
# sin(2*np.pi)를 1 초 (t) 안에 5개 (ff1)의 2*pi 주기의 sin 파를 만들어라.
# 그런데, 1초 안에는 20개의 점을 찍어라.
# np.size(yy)=20

#  compare sine and cosine
yy1= np.cos(2*np.pi*ff1*t1)

########### draw sin and cosine
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t1,yy, 'b.')
# np.size(t1) = 20
# np.size(yy) = 20
plt.xlabel('Time')
plt.ylabel('sin(t)')
plt.grid()
plt.subplot(2,1,2)
plt.plot(t1,yy1,'r*')
plt.xlabel('Time')
plt.ylabel('cos(t)')
plt.grid()



########### Sinusoids (정현파)의 평균값 (mean value) 바꾸기

Fs = 150.0 # sampling rate
Ts = 1.0/Fs # sampling interval
t = np.arange(0,1,Ts) # time vector
ff = 5
# frequency of the signal
y = np.sin(2*np.pi*ff*t)
a0=2
y1= a0+np.sin(2*np.pi*ff*t)
###########
plt.figure(100)
plt.subplot(2,1,1)
plt.plot(t,y, 'b.')
# np.size(t) = 150
# np.size(y) = 150
plt.xlabel('Time')
plt.ylabel('sin(t)')
plt.grid()
plt.subplot(2,1,2)
plt.plot(t,y1,'r*')
plt.xlabel('Time')
plt.ylabel('a0+sin(t)')
plt.grid()


############  Sinusoids (정현파)의 진폭 (Amplitude) 바꾸기
Fs = 150.0 # sampling rate
Ts = 1.0/Fs # sampling interval
t = np.arange(0,1,Ts) # time vector
ff = 5
# frequency of the signal
y = np.sin(2*np.pi*ff*t)
c1=3
y2= c1*np.sin(2*np.pi*ff*t)
###########
plt.figure(101)
plt.subplot(2,1,1)
plt.plot(t,y, 'b.')
# np.size(t) = 150
# np.size(y) = 150
plt.xlabel('Time')
plt.ylabel('sin(t)')
plt.grid()
plt.subplot(2,1,2)
plt.plot(t,y2,'r*')
plt.xlabel('Time')
plt.ylabel('c1*sin(t)')
plt.grid()

############# Sinusoids (정현파)의 주파수 바꾸기
Fs = 150.0 # sampling rate
Ts = 1.0/Fs # sampling interval
t = np.arange(0,1,Ts) # time vector
ff = 5
# frequency of the signal
y = np.sin(2*np.pi*ff*t)
# (sin(2*np.pi) 를 ) 1 초 (t) 안에 5 개 (ff) 의 2*pi 주기의
# sin 파를 만들어라 . 그런데 , 1 초 안에는 150 개의 점을 찍어라 .
# np.size(y)=150
ff1=10
y3 = np.sin(2*np.pi*ff1*t)
# 1 초 (t) 안에 10 개 (ff1) 의 2*pi 주기의 sin 파를
# 만들어라. 그런데 , 1 초 안에는 150 개의 점을 찍어라 .
plt.figure(102)
plt.subplot(2,1,1)
plt.plot(t,y, 'b.-')
# np.size(t) = 150
# np.size(y) = 150
plt.xlabel('Time')
plt.ylabel('sin(2*pi*5*t)')
plt.grid()
plt.subplot(2,1,2)
plt.plot(t,y3,'r*-')
plt.xlabel('Time')
plt.ylabel('sin(2*pi*10*t)')
plt.grid()

##############Sinusoids (정현파)의 위상 바꾸기
Fs = 150.0 # sampling rate
Ts = 1.0/Fs # sampling interval
t = np.arange(0,1,Ts) # time vector
ff = 5
# frequency of the signal
y = np.sin(2*np.pi*ff*t)
theta=np.pi/2
y4 = np.sin(2*np.pi*ff*t+theta)
###########
plt.figure(103)
plt.subplot(2,1,1)
plt.plot(t,y, 'b.-')
plt.xlabel('Time')
plt.ylabel('sin(2*pi*5*t)')
plt.grid()
plt.subplot(2,1,2)
plt.plot(t,y4,'r*-')
plt.xlabel('Time')
plt.ylabel('sin(2*pi*5*t+np.pi/2)')
plt.grid()


############################


import numpy as np
import matplotlib.pyplot as plt

Fs = 150.0
Ts = 1.0/Fs
t = np.arange(0, 1, Ts)
ff1 = 1
c1=1
sqwave1 = (1/c1)*np.sin(c1*2 * np.pi * ff1 * t)
plt.figure(1)
plt.plot(t, sqwave1)

c1=c1+2
sqwave3 = (1/c1)*np.sin(c1*2 * np.pi * ff1 * t)
plt.plot(t, sqwave3)

c1=c1+2
sqwave5 = (1/c1)*np.sin(c1*2 * np.pi * ff1 * t)
plt.plot(t, sqwave5)


#plt.figure(2)

sqwave = sqwave1+sqwave3+sqwave5
plt.plot(t, sqwave)


Fs = 150.0 # sampling rate
Ts = 1.0/Fs # sampling interval
t = np.arange(0, 1, Ts)
# time vector
ff1 = 1 # frequency of the signal
sqwave=0
c1=1
plt.figure(201)

for loop in np.arange(1, 10, 1):
    sqwave = sqwave + (1/c1)*np.sin(c1*2 * np.pi * ff1 * t)
    # fourier series
    plt.plot(t, sqwave)
    c1=c1+2
    plt.pause(3)


#################################


plt.figure(11)
f1=(1/1)*np.sin(1*2 * np.pi * ff1 * t)
plt.plot(t, f1)
f3=(1/3)*np.sin(3*2 * np.pi * ff1 * t)
plt.plot(t, f3)
f5=(1/5)*np.sin(5*2 * np.pi * ff1 * t)
plt.plot(t, f5)
f7=(1/7)*np.sin(7*2 * np.pi * ff1 * t)
plt.plot(t, f7)
f9=(1/9)*np.sin(9*2 * np.pi * ff1 * t)
plt.plot(t, f9)
f11=(1/11)*np.sin(11*2 * np.pi * ff1 * t)
plt.plot(t, f11)
f13=(1/13)*np.sin(13*2 * np.pi * ff1 *t)
plt.plot(t, f13)
f15=(1/15)*np.sin(15*2 * np.pi * ff1 *t)
plt.plot(t, f15)
f17=(1/17)*np.sin(17*2 * np.pi * ff1 *t)
plt.plot(t, f17)


#################################

plt.figure(13)
f1=(1/1)*np.sin(1*2 * np.pi * ff1 * t)
plt.subplot(3, 1, 1)
plt.plot(t, f1)
f3=(1/3)*np.sin(3*2 * np.pi * ff1 * t)
plt.subplot(3, 1, 2)
plt.plot(t, f3)
plt.subplot(3, 1, 3)
plt.plot(t, f1+f3)

################################
plt.figure(14)
f1=(1/1)*np.sin(1*2 * np.pi * ff1 * t)
plt.subplot(4, 1, 1)
plt.plot(t, f1)
f3=(1/3)*np.sin(3*2 * np.pi * ff1 * t)
plt.subplot(4, 1, 2)
plt.plot(t, f3)
f5=(1/5)*np.sin(5*2 * np.pi * ff1 * t)
plt.subplot(4, 1, 3)
plt.plot(t, f5)
plt.subplot(4, 1, 4)
plt.plot(t, f1+f3+f5)


############################

# sin() 홀수 정현파 더하기 --> 사각형파 만들기

import numpy as np
import matplotlib.pyplot as plt

Fs = 150.0  # sampling rate
Ts = 1.0/Fs # sampling interval
t = np.arange(0, 1, Ts)  # time vector
ff1 = 5
sqwave=0
c1=1
plt.figure(301)
for loop in np.arange(1, 10, 1):
    sqwave = sqwave+(1/c1)*np.sin(c1*2*np.pi*ff1*t)
    # fourier series
    plt.plot(t, sqwave)
    c1=c1+2
    plt.pause(3)
    
############################

# sin() 홀수+짝수 정현파 더하기 --> 톱니파 만들기

Fs = 150.0  # sampling rate
Ts = 1.0/Fs # sampling interval
t = np.arange(0, 1, Ts)  # time vector
ff1 = 5
sqwave=0
c1=1
plt.figure(302)
for loop in np.arange(1, 10, 1):
    sqwave = sqwave+(1/c1)*np.sin(c1*2*np.pi*ff1*t)
    # fourier series
    plt.plot(t, sqwave)
    
############################   
# cos() 홀수 정현파 더하기 --> 넓은 삼각형파 만들기


Fs = 150.0  # sampling rate
Ts = 1.0/Fs # sampling interval
t = np.arange(0, 1, Ts)  # time vector
ff1 = 5
sqwave=0
c1=1
plt.figure(303)
for loop in np.arange(1, 10, 1):
    sqwave = sqwave+(1/c1)*np.cos(c1*2*np.pi*ff1*t)
    # fourier series
    plt.plot(t, sqwave)
    c1=c1+2
    plt.pause(3)
    
