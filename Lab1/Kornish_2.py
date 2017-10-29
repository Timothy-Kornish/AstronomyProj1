# loading imports

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.misc

# Loading in the flash drive as the directory and the firts data sets

myDir_1 = '/Volumes/TIM/Lab1_data/'
myData = np.zeros((1,10))
myData_2 = np.zeros ((1,100))

myData = np.loadtxt(myDir_1+'Lab1_n10t10_002')
myData_2 = np.loadtxt(myDir_1+'Lab1_n100t10_020')

#Analyze data

mean_myData =np.mean(myData)
std_myData = np.std(myData)
mean_myData_2 = np.mean(myData_2)
std_myData_2= np.std(myData_2)
mean_data = np.array([mean_myData,mean_myData_2])
sdom= np.std (mean_data)

print mean_myData
print std_myData
print mean_myData_2
print std_myData_2

# Scatter plot for section 2 of the lab
plt.figure (1)

plt.plot(myData, 'ro')
plt.xlabel('Sample Number')
plt.ylabel('Photon $(counts/10ms)$')
plt.show()

plt.figure(2)
plt.plot(myData_2, 'go')
plt.xlabel('Sample number')
plt.ylabel('Photon $(counts/10ms)$')
plt.show()


# Plot for section 3 of the lab

data_3_1 = np.zeros((6,100))
data_3_2 = np.zeros((6,1000))

for i in range (1,7):
    data_3_1[i-1,:] = np.loadtxt(myDir_1+'Lab1_n100t10_02'+str(i))
for j in range (1,7):
    data_3_2[j-1,:] = np.loadtxt(myDir_1+'Lab1_n1000t10_00'+str(j))
    
# analyze data section 3

mean_3_1 = np.mean(data_3_1)
std_3_1 = np.std(data_3_1)
mean_3_2 = np.mean(data_3_2)
std_3_2 = np.std(data_3_2)

# 2x3 plot for section 

plt.figure(3)
plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(wspace=0.35)
plt.subplot(231)
plt.hist(data_3_1[0,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplot(232)
plt.hist(data_3_1[1,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplot(233)
plt.hist(data_3_1[2,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplot(234)
plt.hist(data_3_1[3,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplot(235)
plt.hist(data_3_1[4,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplot(236)
plt.hist(data_3_1[5,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')

plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.show()

# data set 2

plt.figure(4)

plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(wspace=0.45)
plt.subplot(231)
plt.hist(data_3_2[0,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplot(232)
plt.hist(data_3_2[1,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplot(233)
plt.hist(data_3_2[2,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplot(234)
plt.hist(data_3_2[3,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplot(235)
plt.hist(data_3_2[4,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.subplot(236)
plt.hist(data_3_2[5,:],10)
plt.xlabel('Photon counts/sample')
plt.ylabel('samples')

plt.tick_params(axis='both', which='major', labelsize=8)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.show()

# section 3.1
sec_31 = np.zeros((12,100))

for i in range (1,10):
    sec_31[i-1,:] = np.loadtxt(myDir_1 +'Lab1_n100t1_2048_00'+str(i))
for i in range (10,13):
    sec_31[i-1,:] = np.loadtxt(myDir_1 +'Lab1_n100t1_2048_0'+str(i))
        
std_31 = np.zeros(12)
for i in range (1,13):
    std_31[i-1] = np.std(sec_31[i-1])
    
mean_31 = np.zeros(12)
for i in range (1,13):
    mean_31[i-1] = np.mean(sec_31[i-1,:])

plt.figure (5)    
plt.scatter (mean_31,std_31**2)
plt.plot(mean_31,mean_31, '-')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('means')
plt.ylabel('variances')
plt.show()

# section 3.2

data_32_1 = np.zeros((1,1000))
data_32_1 = np.loadtxt(myDir_1 + 'Lab1_n1000t1_001')
mean_32_1 = np.mean(data_32_1)
std_32_1 = np.std(data_32_1)
data_32_2 = np.zeros((1,1000))
data_32_2 = np.loadtxt(myDir_1 + 'Lab1_n1000t1_002')
mean_32_2 = np.mean(data_32_2)
std_32_2 = np.std(data_32_2)

plt.figure(6)
mu = mean_32_1 
x= np.arange(16)
p= mu**x / (scipy.misc.factorial(x) * (math.exp(mu)))

plt.hist(data_32_1,8,normed = True, color = 'b')
plt.step(x,p,color='r')
plt.ylabel('% of photon counts')
plt.xlabel('sample#')
plt.figure(7)
mu_2 = mean_32_2 
x_2= np.arange(90)
sigma =std_32_2
p_2= np.exp(-.5 * ((x_2-mu_2)/sigma)**2)/(sigma *math.sqrt(2*np.pi))

plt.hist(data_32_2,10,normed = True, color = 'b')
plt.plot(x_2, p_2,color='r')

plt.ylabel('% of photon counts')
plt.xlabel('sample#')
plt.show()


# Section 3.3

myDir_3 = '/Volumes/TIM/Lab1_3_3/'
nfiles_33 = 10
nsamples_33_1 = 2
data_33_1 = np.zeros((nfiles_33,nsamples_33_1))
for i in range(1,10):
    data_33_1[i-1,:] = np.loadtxt(myDir_3+'n2t10_00'+str(i)) 
for i in range (10,11):
    data_33_1[i-1,:] = np.loadtxt(myDir_3+'n2t10_0'+str(i)) 
mean_33_1 = np.mean(data_33_1)    
std_33_1 = np.std(data_33_1)

nsamples_33_2 = 4
data_33_2 = np.zeros((nfiles_33,nsamples_33_2))
for i in range (1,10):
    data_33_2[i-1,:] = np.loadtxt(myDir_3+'n4t10_00'+str(i)) 
for i in range (10,11):
    data_33_2[i-1,:] = np.loadtxt(myDir_3+'n4t10_0'+str(i)) 
mean_33_2 = np.mean(data_33_2) 
std_33_2 = np.std(data_33_2)

nsamples_33_3 = 8
data_33_3 = np.zeros((nfiles_33,nsamples_33_3))
for i in range (1,10):
    data_33_3[i-1,:] = np.loadtxt(myDir_3+'n8t10_00'+str(i)) 
for i in range (10,11):
    data_33_3[i-1,:] = np.loadtxt(myDir_3+'n8t10_0'+str(i))
mean_33_3 = np.mean(data_33_3) 
std_33_3 = np.std(data_33_3)

nsamples_33_4 = 16
data_33_4 = np.zeros((nfiles_33,nsamples_33_4))
for i in range (1,10):
    data_33_4[i-1,:] = np.loadtxt(myDir_3+'n16t10_00'+str(i)) 
for i in range (10,11):
    data_33_4[i-1,:] = np.loadtxt(myDir_3+'n16t10_0'+str(i))
mean_33_4 = np.mean(data_33_4) 
std_33_4 = np.std(data_33_4)

nsamples_33_5 = 32
data_33_5 = np.zeros((nfiles_33,nsamples_33_5))
for i in range (1,10):
    data_33_5[i-1,:] = np.loadtxt(myDir_3+'n32t10_00'+str(i)) 
for i in range (10,11):
    data_33_5[i-1,:] = np.loadtxt(myDir_3+'n32t10_0'+str(i))
mean_33_5 = np.mean(data_33_5) 
std_33_5 = np.std(data_33_5)

nsamples_33_6 = 64
data_33_6 = np.zeros((nfiles_33,nsamples_33_6))
for i in range (1,10):
    data_33_6[i-1,:] = np.loadtxt(myDir_3+'n64t10_00'+str(i)) 
for i in range (10,11):
    data_33_6[i-1,:] = np.loadtxt(myDir_3+'n64t10_0'+str(i))
mean_33_6 = np.mean(data_33_6) 
std_33_6 = np.std(data_33_6)
       
nsamples_33_7 = 128
data_33_7 = np.zeros((nfiles_33,nsamples_33_7))
for i in range (1,10):
    data_33_7[i-1,:] = np.loadtxt(myDir_3+'n128t10_00'+str(i)) 
for i in range (10,11):
    data_33_7[i-1,:] = np.loadtxt(myDir_3+'n128t10_0'+str(i)) 
mean_33_7 = np.mean(data_33_7) 
std_33_7 = np.std(data_33_7)
                          
nsamples_33_8 = 256
data_33_8 = np.zeros((nfiles_33,nsamples_33_8))
for i in range (1,10):
    data_33_8[i-1,:] = np.loadtxt(myDir_3+'n256t10_00'+str(i)) 
for i in range (10,11):
    data_33_8[i-1,:] = np.loadtxt(myDir_3+'n256t10_0'+str(i))    
mean_33_8 = np.mean(data_33_8) 
std_33_8 = np.std(data_33_8)
        
nsamples_33_9 = 512
data_33_9 = np.zeros((nfiles_33,nsamples_33_9))
for i in range (1,10):
    data_33_9[i-1,:] = np.loadtxt(myDir_3+'n512t10_00'+str(i)) 
for i in range (10,11):
    data_33_9[i-1,:] = np.loadtxt(myDir_3+'n512t10_0'+str(i))
mean_33_9 = np.mean(data_33_9) 
std_33_9 = np.std(data_33_9)
        
nsamples_33_10 = 1024
data_33_10 = np.zeros((nfiles_33,nsamples_33_10))
for i in range (1,10):
    data_33_10[i-1,:] = np.loadtxt(myDir_3+'n1024t10_00'+str(i)) 
for i in range (10,11):
    data_33_10[i-1,:] = np.loadtxt(myDir_3+'n1024t10_0'+str(i))
mean_33_10 = np.mean(data_33_10) 
std_33_10 = np.std(data_33_10)
        
nsamples_33_11 = 2048
data_33_11 = np.zeros((nfiles_33,nsamples_33_11))
for i in range (1,10):
    data_33_11[i-1,:] = np.loadtxt(myDir_3+'n2048t10_00'+str(i)) 
for i in range (10,11):
    data_33_11[i-1,:] = np.loadtxt(myDir_3+'n2048t10_0'+str(i))
mean_33_11 = np.mean(data_33_11) 
std_33_11 = np.std(data_33_11)

mean_means =np.array ([mean_33_1 , mean_33_2 , mean_33_3 , mean_33_4 , mean_33_5 , mean_33_6 , mean_33_7 , mean_33_8 , mean_33_9 , mean_33_10 , mean_33_11])
means = np.sum(mean_means)/11
nsample = 2**(np.arange(11)+1)

plt.figure(8)
plt.scatter(nsample, mean_means)
plt.xlabel('N samples')
plt.ylabel('Mean of means')

plt.show()

# PLots for section 4 of the lab

myDir_4 = '/Volumes/TIM/Lab1_section4/'
data_41 = np.zeros ((1,20))
data_41 = np.loadtxt (myDir_4 + 'Lab1_4_n20t100_001')

on = data_41[0:10]
off = data_41[10:]

onmean = np.mean(on)
offmean = np.mean(off)


data_41_2 = np.zeros((5,20))
for i in range (11,16):
    data_41_2[i-10:] = np.loadtxt(myDir_4 + 'Lab1_4_n20t100_0'+str(i))

on_2 = data_41_2[:,:10]
off_2 = data_41_2[:,10:]
onmean2 = np.mean(on_2)
offmean2 = np.mean(off_2)
#on = data_41[:,:10]
#off = data_41[:,10:]
#
data_41_3 = np.zeros((5,20))
for i in range (16,21):
    data_41_3[i-15:] = np.loadtxt(myDir_4 + 'Lab1_4_n20t100_0'+str(i))

on_3 = data_41_3[:,:10]
off_3 = data_41_3[:,10:]

onmean3 = np.mean(on_3)
offmean3 = np.mean(off_3)
#

data_41_4 = np.zeros((5,20))
for i in range (21,26):
    data_41_4[i-20:] = np.loadtxt(myDir_4 + 'Lab1_4_n20t100_0'+str(i))

on_4 = data_41_4[:,:10]
off_4 = data_41_4[:,10:]

onmean4 = np.mean(on_4)
offmean4 = np.mean(off_4)
#
data_41_5 = np.zeros((5,20))
for i in range (26,31):
    data_41_5[i-25:] = np.loadtxt(myDir_4 + 'Lab1_4_n20t100_0'+str(i))

on_5 = data_41_5[:,:10]
off_5 = data_41_5[:,10:]

onmean5 = np.mean(on_5)
offmean5 = np.mean(off_5)
#
data_41_6 = np.zeros((5,20))
for i in range (31,36):
    data_41_6[i-30:] = np.loadtxt(myDir_4 + 'Lab1_4_n20t100_0'+str(i))

on_6 = data_41_6[:,:10]
off_6 = data_41_6[:,10:]

onmean6 = np.mean(on_6)
offmean6 = np.mean(off_6)
#
data_41_7 = np.zeros((5,20))
for i in range (36,41):
    data_41_7[i-35:] = np.loadtxt(myDir_4 + 'Lab1_4_n20t100_0'+str(i))

on_7 = data_41_7[:,:10]
off_7 = data_41_7[:,10:]

onmean7 = np.mean(on_7)
offmean7 = np.mean(off_7)
#
bon=np.array([onmean,onmean2,onmean3,onmean4,onmean5,onmean6,onmean7])
boff=np.array([offmean,offmean2,offmean3,offmean4,offmean5,offmean6,offmean7])

bonoff = bon-boff
vom = np.var(bonoff)/7
signal = bonoff.sum()/7
snr = bonoff/np.sqrt(vom)
snr_2 = (bonoff)/np.sqrt((bon+boff))
SNR = np.array([snr_2,snr_2,snr_2,snr_2,snr_2,snr_2,snr_2])

plt.figure (9)
plt.plot (on,'ro')
plt.plot (off,'bx')
plt.plot (on-off ,'gv')


plt.xlabel('Sample #')
plt.ylabel('Counts of Photons')
plt.show ()

plt.figure (10)
plt.plot(bonoff,snr,'bo')
plt.ylabel('Signal to Noise Ratio ($SNR$)')
plt.xlabel('LED brightness ($arbitrary$ $values$)')
plt.xscale('log')
plt.yscale('log')
plt.show()
