import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy import stats

dir1 = '/Volumes/TIM/Lab2/Oct3/'
dir2 = '/Volumes/TIM/Lab2/Oct7/'
dir3 = '/Volumes/TIM/Lab2/CCD/'

A1= fits.getdata(dir2+'oct7_flats-001b.fits')
B1= fits.getdata(dir2+'oct7_flats-002b.fits')
r1 = np.mean(A1)/np.mean(B1)
D1 = B1*r1 - A1
var1 = (np.std(D1)**2)/2

A2= fits.getdata(dir2+'oct7_flats-003b.fits')
B2= fits.getdata(dir2+'oct7_flats-004b.fits')
r2 = np.mean(A2)/np.mean(B2)
D2 = B2*r2 - A2
var2 = (np.std(D2)**2)/2

A3= fits.getdata(dir2+'oct7_flats-005b.fits')
B3= fits.getdata(dir2+'oct7_flats-006b.fits')
r3 = np.mean(A3)/np.mean(B3)
D3 = B3*r3 - A3
var3 = (np.std(D3)**2)/2

A4= fits.getdata(dir2+'oct7_flats-007b.fits')
B4= fits.getdata(dir2+'oct7_flats-008b.fits')
r4 = np.mean(A4)/np.mean(B4)
D4 = B4*r4 - A4
var4 = (np.std(D4)**2)/2

A5= fits.getdata(dir2+'oct7_flats-009b.fits')
B5= fits.getdata(dir2+'oct7_flats-010b.fits')
r5 = np.mean(A5)/np.mean(B5)
D5 = B5*r5 - A5
var5 = (np.std(D5)**2)/2

A6= fits.getdata(dir2+'oct7_flats-011b.fits')
B6= fits.getdata(dir2+'oct7_flats-012b.fits')
r6 = np.mean(A6)/np.mean(B6)
D6 = B6*r6 - A6
var6 = (np.std(D6)**2)/2

A7= fits.getdata(dir2+'oct7_flats-013b.fits')
B7= fits.getdata(dir2+'oct7_flats-014b.fits')
r7 = np.mean(A7)/np.mean(B7)
D7 = B7*r7 - A7
var7 = (np.std(D7)**2)/2

A8= fits.getdata(dir2+'oct7_flats-015b.fits')
B8= fits.getdata(dir2+'oct7_flats-016b.fits')
r8 = np.mean(A8)/np.mean(B8)
D8 = B8*r8 - A8
var8 = (np.std(D8)**2)/2

y= np.array([np.mean(A1),np.mean(A2),np.mean(A4),np.mean(A5),np.mean(A6),np.mean(A7),np.mean(A8)])
x=np.array([var1,var2,var4,var5,var6,var7,var8])
g= y/x

plt.figure(1) 
plt.plot(g)
plt.scatter(x,y)
plt.xlim(12000,15000)
plt.ylim(16000,20000)
plt.xlabel('variance')
plt.ylabel('signal')
plt.show()
print np.mean(g)

plt.figure(2)
