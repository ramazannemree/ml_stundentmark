#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
#datasetin eklenmesi
dataset=pd.read_csv('studentmat.csv',';')

##############################################################################
#preprocessing
dataset=dataset.drop('guardian',axis=1)#ilgisiz colon silinir

##############################################################################
# categoric->numeric
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')

school=dataset.iloc[:,0:1].values
school=ohe.fit_transform(school).toarray()
school=pd.DataFrame(school[:,:1],range(395),['school'])

sex=dataset.iloc[:,1:2].values
sex=ohe.fit_transform(sex).toarray()
sex=pd.DataFrame(sex[:,:1],range(395),['sex'])

address=dataset.iloc[:,3:4].values
address=ohe.fit_transform(address).toarray()
address=pd.DataFrame(address[:,:1],range(395),['address'])

famsize=dataset.iloc[:,4:5].values
famsize=ohe.fit_transform(famsize).toarray()
famsize=pd.DataFrame(famsize[:,:1],range(395),['famsize'])

pstaus=dataset.iloc[:,5:6].values
pstaus=ohe.fit_transform(pstaus).toarray()
pstaus=pd.DataFrame(pstaus[:,:1],range(395),['pstaus'])

mjob=dataset.iloc[:,8:9].values
mjob=ohe.fit_transform(mjob).toarray()
mjob=pd.DataFrame(mjob,range(395),['at_home_mjob','healt_mjob','other_mjob','sevices_mjob','teacher_mjob'])

fjob=dataset.iloc[:,9:10].values
fjob=ohe.fit_transform(fjob).toarray()
fjob=pd.DataFrame(fjob,range(395),['at_home_fjob','healt_fjob','other_fjob','sevices_fjob','teacher_fjob'])

reason=dataset.iloc[:,10:11].values
reason=ohe.fit_transform(reason).toarray()
reason=pd.DataFrame(reason,range(395),['course','other_reason','home','reputation'])

schoolsup=dataset.iloc[:,14:15].values
schoolsup=ohe.fit_transform(schoolsup).toarray()
schoolsup=pd.DataFrame(schoolsup[:,:1],range(395),['schoolsup'])

famsup=dataset.iloc[:,15:16].values
famsup=ohe.fit_transform(famsup).toarray()
famsup=pd.DataFrame(famsup[:,:1],range(395),['famsup'])

paid=dataset.iloc[:,16:17].values
paid=ohe.fit_transform(paid).toarray()
paid=pd.DataFrame(paid[:,:1],range(395),['paid'])

activities=dataset.iloc[:,17:18].values
activities=ohe.fit_transform(activities).toarray()
activities=pd.DataFrame(activities[:,:1],range(395),['activities'])

nursery=dataset.iloc[:,18:19].values
nursery=ohe.fit_transform(nursery).toarray()
nursery=pd.DataFrame(nursery[:,:1],range(395),['nursery'])

higher=dataset.iloc[:,19:20].values
higher=ohe.fit_transform(higher).toarray()
higher=pd.DataFrame(higher[:,:1],range(395),['higher'])

internet=dataset.iloc[:,20:21].values
internet=ohe.fit_transform(internet).toarray()
internet=pd.DataFrame(internet[:,:1],range(395),['internet'])

romantic=dataset.iloc[:,21:22].values
romantic=ohe.fit_transform(romantic).toarray()
romantic=pd.DataFrame(romantic[:,:1],range(395),['romantic'])

output=dataset[['G3']]#hedef output
##############################################################################

# numerice çevirilen kolonlar datasetten çıkarılır
dataset=dataset.drop(['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'],axis=1)

# numerice çevirilen colonlar ve dataset birleştirilir
data_end=pd.concat([school,sex,address,famsize,pstaus,mjob,fjob,reason,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,dataset],axis=1)
data_end=data_end.drop('G3',axis=1)

##############################################################################
# son dataset train ve test olarak bölünür
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(data_end,output,test_size=0.20)

# LinearRegression
# sklearn ile
from sklearn.linear_model import LinearRegression
r=LinearRegression()
r.fit(x_train,y_train)
pred=r.predict(x_test)

##############################################################################
# statsmodels ile
import statsmodels.formula.api as sm
# summary tablosunda değişkenlerin p değerine bakarak backward elimination yaptık.
# p-value si en yüksek bağımsız değişkeni modelden çıkardık.
# (0-41 kolonların hepsi yazılıydı tek tek silerek gittim aradaki adımları yazmadım!)
x_l=data_end.iloc[:,[22,27,33,37,39,40,41]].values
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())
#mesela burda 37. kolonun p değeri en yüksek çıktığı için 37 yi kaldırdık
x_l=data_end.iloc[:,[22,27,33,39,40,41]].values
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

x_l=data_end.iloc[:,[27,33,39,40,41]].values 
x=pd.concat([data_end,output],axis=1)
r_ols=sm.ols('output~x_l',x).fit()
print(r_ols.summary())

#p değerlerinin hepsi 0.05 in altında olduğu için elemeyi bitirdik ve tekrar tahmin yapıyoruz.
x2_train, x2_test, y2_train, y2_test=train_test_split(x_l,output,test_size=0.20)
r2=LinearRegression()
r2.fit(x2_train,y2_train)
pred2=r2.predict(x2_test)
