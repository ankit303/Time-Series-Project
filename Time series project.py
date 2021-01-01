#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
random.seed(100)


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
from pandas import read_csv
from matplotlib import pyplot
from numpy import polyfit


# In[3]:


import math
from scipy.stats import norm, chi2


# In[4]:


link='https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
data=pd.read_csv(link)


# In[5]:


data.shape


# In[6]:


type(data)


# In[7]:


print(data.head(5))
print(data.tail(5))
print(data.iloc[1459:1461]) #location of  the missing value
print(data.iloc[2919:2921]) #location of  the missing value 


# In[8]:


Time=['%.3f'%(1981+i/365) for i in range(365)]
for i in range(1982,1991):
  if(i%4==0):
    Time+=['%.3f'%(i+j/366) for j in range(366)]
  else:
    Time+=['%.3f'%(i+j/365) for j in range(365)] 


# In[9]:


mis1=pd.DataFrame({'Date':'1984-12-31','Temp':float('NaN')},index=[1460])
mis2=pd.DataFrame({'Date':'1988-12-31','Temp':float('NaN')},index=[2921])
df=pd.concat([data.iloc[:1460],mis1,data.iloc[1460:2920],mis2,data.iloc[2920:]])
#d=pd.DataFrame(df,index)
df.shape


# In[10]:


df.reset_index(inplace=True)


# In[11]:


df['Time']=Time


# In[12]:


data=df.iloc[:,1:4]


# In[13]:


data.iloc[1458:1463]


# In[14]:


data.iloc[2918:2923]


# In[15]:



train=data.iloc[:3287,:]
test=data.iloc[3287:,:]
print(train.shape)
print(test.shape)


# In[16]:


y1=list(data.iloc[1450:1460,1])+list(data.iloc[1460:1471,1])
y2=list(data.iloc[2911:2921,1])+list(data.iloc[2921:2932,1])


# In[17]:


plt.plot(list(range(1,22)),y1);


# In[18]:


plt.plot(list(range(1,22)),y2);


# In[19]:


a=np.array([364,729,1094,1460,1825,2190,2555,2921,3286,3651])
y=data.iloc[a,1].values
miss1 = np.mean( np.concatenate((y[0:3],y[4:7])) )
miss2 = np.mean( np.concatenate((y[4:7],y[8:11])) )
print(miss1,miss2)
data.iloc[1460,1] = miss1
data.iloc[2921,1] = miss2
y1=list(data.iloc[1450:1460,1])+list(data.iloc[1460:1471,1])
y2=list(data.iloc[2911:2921,1])+list(data.iloc[2921:2932,1])
plt.plot(list(range(1,22)),y1);


# In[20]:


print(data.head(5))


# In[21]:


len(data)


# In[22]:


data.iloc[:,2:3]=[1+i%3652 for i in range(0, len(data))]


# In[23]:


data.iloc[:,2:3]


# In[24]:


data.shape


# In[25]:


train=data.iloc[:2922,:]
test=data.iloc[2922:,:]
print(train.shape,test.shape)


# In[26]:


temp=train.iloc[:,1:2]
n=temp.shape[0]
print(temp)


# In[27]:


temp=[temp.iloc[i] for i in range(n)]


# In[28]:


print(temp[1:5])
print(len(temp))


# In[29]:



def relative_ordering(temp,alpha):
  Q=0
  n=len(temp)
  for i in range(n):
    for j in range(i+1,n):
      if(temp[i]-temp[j]>0):
        Q+=1
  print('Q=',Q)
  T=1-4*Q/(n*(n-1))
  VT=2*(2*n+5)/(9*n*(n-1))
  Z=T/math.sqrt(VT)
  t=norm.ppf(1-alpha/2)
  print('Z=',Z)
  print('t_alpha/2=',t)
  if(abs(Z)<=t):
    print('So, Trend is not present')
  else:
    print('Trend is present')  



# In[30]:


temp=[float(i) for i in temp]


# In[31]:


relative_ordering(temp,0.05)


# In[32]:


def friedman(temp,r_val,c,alpha):
  r=[]
  for i in range(1,9):
    if(i<5):
      year=temp[365*(i-1):i*365]
    else:
      year=temp[365*(i-1)+1:i*365+1]
    year=np.array(year)
    ran=pd.Series(year)
    ran=ran.rank()
    r.append(ran) 
  s=np.array(sum(r))  
  m=c*(r_val+1)/2
  v=c*r_val*(r_val+1) 
  X=12*sum((s-m)**2)/v
  print('X=',X)
  chi_square=chi2.ppf(1-alpha,r_val-1)
  print('Chi square value ',chi_square)
  if(X>chi_square):
    print('Seasonality is present')
  if(X<=chi_square):
    print('No Seasonality')


# In[33]:


friedman(temp,365,8,0.05)


# In[34]:


type(temp)


# In[35]:


def turning_point(temp,alpha):
  P=0
  n=len(temp)
  for i in range(1,n-1):
    if((temp[i]>temp[i-1] and temp[i]>temp[i+1]) or (temp[i]<temp[i-1] and temp[i]<temp[i+1])):
      P+=1
  print('P=',P)
  EP=2*(n-2)/3
  VP=(16*n-29)/90
  Z=(P-EP)/math.sqrt(VP)
  print("Z=",Z)
  t=norm.ppf(1-alpha/2)
  print('t_alpha/2=',t)
  if(abs(Z)>t):
    print('Series is not purely Random')
  else:
    print('Series is purely Random')  



# In[36]:


turning_point(temp,0.05)


# In[39]:



x=[]
# fit polynomial: x^2*b1 + x*b2 + ... + bn
for i in range(1981,1989):
  if(i%4==0):
    for j in range(0,366):
        x.append(j)
  else:
    for j in range(0,365):
        x.append(j)

print(len(x))
y = temp
degree = 4
coef = polyfit(x, y, degree)
print('Coefficients: %s' % coef)
# create curve
curve = list()
for i in range(len(x)):
	value = coef[-1]
	for d in range(degree):
		value += x[i]**(degree-d) * coef[d]
	curve.append(value)


# In[40]:


# plot curve over original data
pyplot.plot(data.Temp[1:2922])
pyplot.plot(curve, color='red', linewidth=3)
pyplot.show()


# In[41]:


# create seasonally adjusted
values = y
diff = list()
for i in range(len(values)):
	value = values[i] - curve[i]
	diff.append(value)
pyplot.plot(diff)
pyplot.show()


# In[42]:


print(diff[0:5]) # temperature after removing seasonality (first six observation)


# In[43]:


print(curve[0:5])   #  estimates of seasonality (first six observation)


# In[44]:


friedman(diff,365,8,0.05)


# In[45]:


turning_point(diff,0.05)


# In[46]:


train.shape
print(train.head())


# In[47]:


train.iloc[:,2]=x


# In[48]:


train.iloc[:,1:2]=diff


# In[49]:


print(train.head())


# In[50]:


#  Checking for stationarity


# In[51]:


pyplot.plot(diff)
pyplot.show()
# in plot seems to be stationary as no trend and seasonal component seems here.


# In[52]:


import random  
from random import sample 
#  drawing random sample from diff and checking mean and var


# In[53]:


s1=diff[0:1495]
s2=diff[1495:(n-1)]


# In[54]:


print(s1[0:5])


# In[56]:


# checking equality of mean and var

mean1, mean2 = (sum(s1))/(len(s1)),(sum(s2))/(len(s2))
def vari(list):
    mean=sum(list)/len(list)
    summ=0
    for i in list:
        summ=summ+(i-mean)**2
    variance=summ/len(list)
    return(variance)
 
print('mean1=%f, mean2=%f' % (mean1,mean2))
var1,var2=vari(s1),vari(s2)
print('variance1=%f, variance2=%f' % ((var1), var2))

# so mean and var are very close so data seems stationarity


# In[57]:


#Augmented Dickey-Fuller test  for testing stationarity
from statsmodels.tsa.stattools import adfuller
X = diff
result = adfuller(X,autolag='AIC')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Lag Used: %f' % result[2])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
if result[1]<0.05:
    print("data series is stationary")
else:
    print("data serpip install pmdarimaies is not stationary")


# In[58]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(diff,lags=30)
plot_pacf(diff,lags=30)


# From ACF plot lag q=9, from  PACF plot and ADF test lag p =6

# In[59]:


pip install pmdarima


# In[60]:


from pmdarima.arima import auto_arima


# In[61]:


#stepwise_fit = auto_arima(diff , trace =True ,suppress_warninigs =True)
#stepwise_fit.summary()


# In[62]:


from statsmodels.tsa.arima_model import ARIMA
train.shape
train.head(5)
#test.shape
model = ARIMA(diff,order=(6,1,9))
model = model.fit()
model.summary()


# In[63]:


start =len(train)
end = len(train)+len(test)-1
pred = model.predict(start=start ,end = end ,typ='levels')
print(pred)


# In[64]:


s_comp=curve[0:730]


# In[65]:


len(s_comp)


# In[66]:


pred1=pred+s_comp


# In[67]:



test = list()
for i in range(730):
	test1 = data.Temp[2922+i]
	test.append(test1)



pyplot.plot(test,linewidth=1)
pyplot.plot(pred1, color='red', linewidth=3)
pyplot.show()


# In[68]:


len(test)


# In[70]:


sum1=0

for i in range(len(test)):
    sum1=sum1+(test[i]-pred1[i])**2
    

print("sum=",sum1)
RMSE=math.sqrt(sum1/len(test))
print("RMSE=",RMSE)

