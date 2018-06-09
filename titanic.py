# -*- coding: utf-8 -*-
"""
@author: Franco Reynaldo Condori Choque 

"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from   sklearn.neural_network import MLPClassifier


#Para recodificar variables
def recodificar(col, nuevo_codigo):
  col_cod = pd.Series(col, copy=True)
  for llave, valor in nuevo_codigo.items():
    col_cod.replace(llave, valor, inplace=True)
  return col_cod

def open_close_plot():
    plt.show()
    plt.close()


##1)

os.chdir("D:/")
titanic = pd.read_csv("train.csv",sep=",",decimal=".",index_col=0)
respuesta = pd.read_csv("test.csv",sep=",",decimal=".",index_col=0)

#dimension de la tabla
titanic.shape
respuesta.shape
#cabesal
titanic.head(0)
respuesta.head(0)
#Quitando valores unicos
titanic=titanic.iloc[:,[0,1,3,4,5,6,10]]
respuesta=respuesta.iloc[:,[0,2,3,4,5,9]]

#comparando la cantidad de los datos que no contienen datos faltantes
titanic.shape
respuesta.shape

respuesta.dropna().shape
respuesta.shape

respuesta.describe()
titanic.describe()

titanic.info()
respuesta.info()
#Variable categorica
titanic['Pclass'].dropna().shape
titanic['Pclass'].shape
titanic['Pclass']=titanic['Pclass'].astype('category')

respuesta['Pclass'].dropna().shape
respuesta['Pclass'].shape
respuesta['Pclass']=respuesta['Pclass'].astype('category')


titanic=titanic.fillna(titanic.mean())
respuesta=respuesta.fillna(respuesta.mean())
#respuesta=respuesta.fillna(respuesta.mean())



#Categorizando Variables para los modelos


titanic['Embarked'].dropna().shape
titanic['Embarked'].shape
titanic["Embarked"].value_counts()
titanic["Embarked"].fillna('S', inplace=True)
print(titanic["Embarked"].value_counts())
titanic["Embarked"]= recodificar(titanic["Embarked"], {'S':1,'C':2,'Q':3})
print(titanic["Embarked"].value_counts())
titanic['Embarked']=titanic['Embarked'].astype('category')

respuesta['Embarked'].dropna().shape
respuesta['Embarked'].shape
respuesta["Embarked"].value_counts()
respuesta["Embarked"].fillna('S', inplace=True)
print(respuesta["Embarked"].value_counts())
respuesta["Embarked"]= recodificar(respuesta["Embarked"], {'S':1,'C':2,'Q':3})
print(respuesta["Embarked"].value_counts())
respuesta['Embarked']=respuesta['Embarked'].astype('category')

######


titanic['Sex'].dropna().shape
titanic['Sex'].shape
print(titanic["Sex"].value_counts())
titanic["Sex"]= recodificar(titanic["Sex"], {'male':1,'female':0})
print(titanic["Sex"].value_counts())
titanic['Sex']=titanic['Sex'].astype('category')

respuesta['Sex'].dropna().shape
respuesta['Sex'].shape
print(respuesta["Sex"].value_counts())
respuesta["Sex"]= recodificar(respuesta["Sex"], {'male':1,'female':0})
print(respuesta["Sex"].value_counts())
respuesta['Sex']=respuesta['Sex'].astype('category')


titanic['Survived']=titanic['Survived'].astype('category')
print(titanic.info())





titanic['Age'].dropna().shape
titanic['Age'].shape



plt.hist(titanic['Age'].dropna())


plt.hist(respuesta['Age'].dropna())



###Separando la variable a predecir,

X=titanic.iloc[:,[1,2,3,4,5,6]]
y=titanic.iloc[:,0:1]
print(X)
print(y)

A=respuesta.iloc[:,[0,1,2,3,4,5]]
y=respuesta.iloc[:,0:1]
print(X)
print(y)


#####################################TAREA################################################
instancia_red = MLPClassifier(solver='lbfgs', random_state=0)
instancia_red.fit(X,y)

x=instancia_red.predict(A)
y=pd.DataFrame(x)
y.to_csv('titanic_f.csv')
y.shape
y.to_csv('holaprueba.csv')

################################################################################




