# -*- coding: utf-8 -*-
"""
@author: Franco Reynaldo Condori Choque - Tarea 8-Ciencia de Datos con Python

"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
#import graphviz
os.environ["PATH"]+=os.pathsep+'C:/Program Files (x86)/Graphviz2.38/bin'
import matplotlib.image as mpimg
from   sklearn.tree import DecisionTreeClassifier
from   sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from   sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import patsy
import seaborn

#Para recodificar variables
def recodificar(col, nuevo_codigo):
  col_cod = pd.Series(col, copy=True)
  for llave, valor in nuevo_codigo.items():
    col_cod.replace(llave, valor, inplace=True)
  return col_cod

def open_close_plot():
    plt.show()
    plt.close()
    
#Graficando el arbol
def graficar_arbol(grafico = None):
    grafico.format = "png"
    archivo  = grafico.render()
    img = mpimg.imread(archivo)
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.close()

# Ejercicio 2

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

#Eliminando Registros con Datos Faltantes
#titanic=titanic.dropna()
#respuesta=respuesta.dropna().shape

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


#Variable a predecir
#print(titanic["Survived"].value_counts())
#titanic["Survived"]= recodificar(titanic["Survived"], {1:"Sobrevivio",0:"NoSobrevivio"})
#print(titanic["Survived"].value_counts())

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

### Particion de la Tabla.

#Partiendo las tablas a un 25 % para testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

###El 25% de la tabla de testing

print(X_test)
print(y_test)
#####################################################################






#########################################################################
##3)

###K-VECINOS



#Mediante el cosntructor inicializa el atributo n_neighbors=3
instancia_knn = KNeighborsClassifier(n_neighbors=3)

#Entrena el modelo llamando al metodo fit
instancia_knn.fit(X_train,y_train)

#Imprime las predicciones en testing
print("Las predicciones en Testing son: {}".format(instancia_knn.predict(X_test)))




#Porcentaje de prediccion global
print("Precision en Testing: {:.2f}".format(instancia_knn.score(X_test, y_test)))

#Matriz de Confusion
prediccion = instancia_knn.predict(X_test)
MC_Kvecinos = confusion_matrix(y_test, prediccion)
print("Matriz de Confusion:\n{}".format(MC_Kvecinos))

###ARBOLES

#Mediante el constructor inicializa la instancia_arbol
instancia_arbol = DecisionTreeClassifier(random_state=0,min_samples_leaf=1500)

#Entrena el modelo llamando al metodo fit
instancia_arbol.fit(X_train,y_train)

#Imprime las predicciones en testing
print("Las predicciones en Testing son: {}".format(instancia_arbol.predict(X_test)))

#Porcentaje de prediccion global
print("Precision en Testing: {:.3f}".format(instancia_arbol.score(X_test, y_test)))

#Matriz de confusion
prediccion = instancia_arbol.predict(X_test)
MC_Arboles = confusion_matrix(y_test, prediccion)
print("Matriz de Confusion:\n{}".format(MC_Arboles))

###BOSQUES ALEATORIOS

#Mediante el constructor inicializa la instancia_bosque
instancia_bosque = RandomForestClassifier(n_estimators=10, random_state=0)

#Entrena el modelo llamando al metodo fit
instancia_bosque.fit(X_train,y_train)

#Imprime las predicciones en testing
print("Las predicciones en Testing son: {}".format(instancia_bosque.predict(X_test)))

#Porcentaje de prediccion global
print("Precision en Testing: {:.3f}".format(instancia_bosque.score(X_test, y_test)))

#Matriz de confusion
prediccion = instancia_bosque.predict(X_test)
MC_Bosques = confusion_matrix(y_test, prediccion)
print("Matriz de Confusion:\n{}".format(MC_Bosques))


###POTENCIACION

#Mediante el constructor inicializa la instancia_potenciacion
instancia_potenciacion = GradientBoostingClassifier(n_estimators=10, random_state=0)

#Entrena el modelo llamando al metodo fit
instancia_potenciacion.fit(X_train,y_train)

#Imprime las predicciones en testing
print("Las predicciones en Testing son: {}".format(instancia_potenciacion.predict(X_test)))

#Porcentaje de prediccion global
print("Precision en Testing: {:.3f}".format(instancia_potenciacion.score(X_test, y_test)))

#Matriz de confusion
prediccion = instancia_potenciacion.predict(X_test)
MC_Potenciacion = confusion_matrix(y_test, prediccion)
print("Matriz de Confusion:\n{}".format(MC_Potenciacion))


###SVM

#Mediante el constructor inicializa la instancia_svm
instancia_svm = SVC()

#Entrena el modelo llamando al metodo fit
instancia_svm.fit(X_train,y_train)

#Imprime las predicciones en testing
print("Las predicciones en Testing son: {}".format(instancia_svm.predict(X_test)))

#Porcentaje de prediccion global
print("Precision en Testing: {:.3f}".format(instancia_svm.score(X_test, y_test)))

#Matriz de confusion
prediccion = instancia_svm.predict(X_test)
MC_svm = confusion_matrix(y_test, prediccion)
print("Matriz de Confusion:\n{}".format(MC_svm))


###REDES NEURONALES


#Mediante el constructor inicializa la instancia_red
instancia_red = MLPClassifier(solver='lbfgs', random_state=0)

#Entrena el modelo llamando al metodo fit
instancia_red.fit(X_train,y_train)

#Imprime las predicciones en testing
print("Las predicciones en Testing son: {}".format(instancia_red.predict(X_test)))

#Porcentaje de prediccion global
print("Precision en Testing: {:.3f}".format(instancia_red.score(X_test, y_test)))

#Matriz de confusion
prediccion = instancia_red.predict(X_test)
MC_red = confusion_matrix(y_test, prediccion)
print("Matriz de Confusion:\n{}".format(MC_red))


#####################################TAREA################################################
instancia_red = MLPClassifier(solver='lbfgs', random_state=0)
instancia_red.fit(X_train,y_train)

x=instancia_knn.predict(A)
y=pd.DataFrame(x)
y.to_csv('titanic_f.csv')
y.shape
y_test.to_csv('holaprueba.csv')

################################################################################



##4)

#def indices_general(MC, nombres = None):
#    precision_global = MC.diagonal().sum() / MC.sum()
#    error_global = 1 - precision_global
#    precision_positiva= MC[1,1]/MC[1,:].sum()
#    precision_negativa= MC[0,0]/MC[0,:].sum()
#    falsos_positivos= MC[0,1]/MC[0,:].sum()
#    falsos_negativos=MC[1,0]/MC[1,:].sum()
#    acertividad_positiva=MC[1,1]/MC[:,1].sum()
#    acertividad_negativa= MC[0,0]/MC[:,0].sum()
#    presicion_categoria  = pd.DataFrame(MC.diagonal()/sum(MC.T)).T
#    if nombres!=None:
#        presicion_categoria.columns = nombres
#    return {"Matriz de Confusion":MC,
#            "Precision Global":precision_global,
#            "Error Global":error_global,
#            'Precision Positica' : precision_positiva,
#            'Precision Negativa' : precision_negativa,
#            'Falsos Positivos' : falsos_positivos,
#            'Falsos Negativos' : falsos_negativos,
#            'Acertividad Positiva' : acertividad_positiva,
#            'Acertividad Negativa' : acertividad_negativa,
#            "Precision por categoria":presicion_categoria}
#
#### INDICES DE K vECINOS
#
#
#
##Índices de Calidad de los Modelo
#indices_kvecinos = indices_general(MC_Kvecinos)
#for k in indices_kvecinos:
#    print("\n%s:\n%s"%(k,str(indices_kvecinos[k])))
#
####INDICES DE ARBOLES
#
#indices_arboles = indices_general(MC_Arboles)
#for k in indices_arboles:
#    print("\n%s:\n%s"%(k,str(indices_arboles[k])))
#
#
####INDICES DE BOSQUES
#
#
#indices_bosques = indices_general(MC_Bosques)
#for k in indices_bosques:
#    print("\n%s:\n%s"%(k,str(indices_bosques[k])))
#
#
####INDICES DE POTENCIACION
#
#
#indices_potenciacion = indices_general(MC_Potenciacion)
#for k in indices_potenciacion:
#    print("\n%s:\n%s"%(k,str(indices_potenciacion[k])))
#
#
####INDICES DE SVM
#
#
#indices_svm = indices_general(MC_svm)
#for k in indices_svm:
#    print("\n%s:\n%s"%(k,str(indices_svm[k])))
#
#
####INDICES DE REDES NEURONALES
#
#
#indices_red = indices_general(MC_red)
#for k in indices_red:
#    print("\n%s:\n%s"%(k,str(indices_red[k])))
#
####TABLA COMPARATIVA DE TODOS LOS MODELOS
#
#PrecisionGlobal=[]
#ErrorGlobal=[]
#PrecisionPositiva=[]
#PrecisionNegativa=[]
#FalsosPositivos=[]
#FalsosNegativos=[]
#AcertividadPositiva=[]
#AcertividadNegativa=[]
#
#for i in [MC_Kvecinos,MC_Arboles,MC_Bosques,MC_Potenciacion,MC_svm,MC_red]:
#    PrecisionGlobal.append(indices_general(i)['Precision Global'])
#    ErrorGlobal.append(indices_general(i)['Error Global'])
#    PrecisionPositiva.append(indices_general(i)['Precision Positica'])
#    PrecisionNegativa.append(indices_general(i)['Precision Negativa'])
#    FalsosPositivos.append(indices_general(i)['Falsos Positivos'])
#    FalsosNegativos.append(indices_general(i)['Falsos Negativos'])
#    AcertividadPositiva.append(indices_general(i)['Acertividad Positiva'])
#    AcertividadNegativa.append(indices_general(i)['Acertividad Negativa'])
#
#
#vectores=[PrecisionGlobal,ErrorGlobal,PrecisionPositiva,PrecisionNegativa,FalsosPositivos,FalsosNegativos,AcertividadPositiva,AcertividadNegativa]
#
#N=pd.DataFrame({'PrecisionGlobal':PrecisionGlobal,'ErrorGlobal':ErrorGlobal,'PrecisionPositiva':PrecisionPositiva,'PrecisionNegativa':PrecisionNegativa,'FalsosPositivos':FalsosPositivos,'FalsosNegativos':FalsosNegativos,'AcertividadPositiva':AcertividadPositiva,'AcertividadNegativa':AcertividadNegativa},index=['MC_Kvecinos','MC_Arboles','MC_Bosques','MC_Potenciacion','MC_svm','MC_red'])
#print(N)
#

### **RESPUESTA:** El mejor metodo es POTENCIACION.

###5)
#modelos=[KNeighborsClassifier(n_neighbors=3),
#DecisionTreeClassifier(random_state=0,min_samples_leaf=1500),
#RandomForestClassifier(n_estimators=10, random_state=0),
#GradientBoostingClassifier(n_estimators=10, random_state=0),
#SVC(),
#MLPClassifier(solver='lbfgs', random_state=0)]
#
#modelos_nombres=['KVECINOS','ARBOLES','BOSQUES','POTENCIASION','SVM','REDES_NEURONALES']
#
##Validacion Cruzada con 5 grupos.
#
#instancia_kfold = KFold(n_splits=5)
#j=0
#for i in modelos:
#  porcentajes = cross_val_score(i, X, y.iloc[:,0].values, cv=instancia_kfold)
#  print(modelos_nombres[j])
#  print("Porcentaje de deteccion por grupo:\n{}".format(porcentajes))
#  print("Promedio de deteccion: {:.2f}".format(porcentajes.mean()))
#  j=j+1


### **RESPUESTA:** El mejor metodo sigue siendo POTENCIACION..

##6)

### Curva ROC para los modelos.

nuevo_y = []
for i in range(len(y_test)):
    #print(y_test.iloc[i])
    if y_test.iloc[i] == 1:
        nuevo_y.append(1)
    else:
        nuevo_y.append(0)
print(y_test)
print(nuevo_y)

pr_n_instancia_vecinos, pr_p_instancia_vecinos, umbral_cero_instancia_vecinos = roc_curve(nuevo_y, instancia_knn.predict_proba(X_test)[:, 1])
pr_n_instancia_arbol, pr_p_instancia_arbol, umbral_cero_instancia_arbol = roc_curve(nuevo_y, instancia_arbol.predict_proba(X_test)[:, 1])
pr_n_instancia_bosque, pr_p_instancia_bosque, umbral_cero_instancia_bosque = roc_curve(nuevo_y, instancia_bosque.predict_proba(X_test)[:, 1])
pr_n_instancia_potenciasion, pr_p_instancia_potenciasion, umbral_cero_instancia_potenciasion = roc_curve(nuevo_y, instancia_potenciacion.predict_proba(X_test)[:, 1])
pr_n, pr_p, umbral = roc_curve(nuevo_y, instancia_svm.fit(X_train,y_train).decision_function(X_test))
pr_n_instancia_red, pr_p_instancia_red, umbral_cero_instancia_red = roc_curve(nuevo_y, instancia_red.predict_proba(X_test)[:, 1])


plt.plot(pr_n_instancia_vecinos, pr_p_instancia_vecinos, label="Curva ROC - K Vecinos")
plt.plot(pr_n_instancia_arbol, pr_p_instancia_arbol, label="Curva ROC - Arboles")
plt.plot(pr_n_instancia_bosque, pr_p_instancia_bosque, label="Curva ROC - Bosques Aleatorios")
plt.plot(pr_n_instancia_potenciasion, pr_p_instancia_potenciasion, label="Curva ROC - Potenciacion")
plt.plot(pr_n, pr_p, label="Curba ROC - SVM")
plt.plot(pr_n_instancia_red, pr_p_instancia_red, label="Curva ROC - Redes Neuronales")
plt.xlabel("Precision Negativa")
plt.ylabel("Presicion Positiva")
plt.legend(loc=4)
open_close_plot()

### Calculo de Area

kvecinos_area = roc_auc_score(nuevo_y,instancia_knn.predict_proba(X_test)[:, 1])
arboles_area = roc_auc_score(nuevo_y, instancia_arbol.predict_proba(X_test)[:, 1])
bosques_area = roc_auc_score(nuevo_y, instancia_bosque.predict_proba(X_test)[:, 1])
potenciacion_area = roc_auc_score(nuevo_y,instancia_potenciacion.predict_proba(X_test)[:, 1])
svm_area = roc_auc_score(nuevo_y, instancia_svm.fit(X_train,y_train).decision_function(X_test))
redes_area = roc_auc_score(nuevo_y, instancia_red.predict_proba(X_test)[:, 1])
print("Area bajo la curva ROC en K Vecinos: {:.3f}".format(kvecinos_area))
print("Area bajo la curva ROC en Arboles: {:.3f}".format(arboles_area))
print("Area bajo la curva ROC en Bosques Aleatorios: {:.3f}".format(bosques_area))
print("Area bajo la curva ROC en Potenciacion: {:.3f}".format(potenciacion_area))
print("Area bajo la curva ROC en SVM: {:.3f}".format(svm_area))
print("Area bajo la curva ROC en Redes Neuronales: {:.3f}".format(redes_area))


### **RESPUESTA:** El mejor con la curva ROC sigue siendo POTENCIACION.

