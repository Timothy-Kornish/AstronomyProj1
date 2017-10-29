
# importing packages 

from astropy.time import Time, TimeDelta
import time
import math
import matplotlib.pyplot as plt
import numpy as np

# Creating a variable array, variable name is arbitrary

a = np.arange(1000)

# acquiring input of period from user to create variables

period_days = input("Planets period in days " )
epochJd = input("planets epoch in Julian Days ")

# Calculating the transit times 

trans = epochJd + a*period_days

trans_time = Time(trans, format = 'jd', scale = 'utc')

# Accounting for daylight savings issue and converting into Mountain Daylight Time (MDT)

if time.daylight == 1:
    trans_MDT = trans_time - TimeDelta(21600, format ='sec') # 21600 is in seconds = 6 hours 
else:
    trans_MDT = trans_time - TimeDelta(25200, format ='sec') # 25200 is in seconds = 7 hours
 
Trans_MDT = Time(trans_MDT, out_subfmt = 'date_hm').iso

# Getting the transits between September 16th, 7 pm, 2014 to October 1st, 7 am, 2014

trans_range = (trans_MDT > '2014-09-16 19:00') & (trans_MDT < '2014-10-01 07:00')


transits = trans_MDT[trans_range]

# converting the output in UT time instead of Julian Days

tran=Time(transits, out_subfmt = 'date_hm').iso

# printing transits and there time of transit 

print "The time of transits are in MDT are (year-month-day hour:minute): ", tran
