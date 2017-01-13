# -*- coding: utf-8 -*-

"""
Created on Thu Dec  8 19:53:15 2016

@author: Côme Girschig, Morgan Raffray & Ninon Tardivel
"""

import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
from scipy.stats import shapiro

#H: hauteur de la couche limite au dessus de laquelle le vent s'annule
#z: altitude de la particule
#alpha: coefficient de vent


#Définitions de différents profils de vent verticaux
def vent_constant(z, H, alpha):
    if z > H or z < 0:
        return 0
    else:
        return alpha

def vent_lineaire(z, H, alpha):
    if z > H or z < 0:
        return 0
    else:
        return alpha * z
    
    #Création profils de vent sinusoidaux, le premier toujours positif le second selon fonction sinus       
def vent_sin_pos(z, H, alpha): #vent uniquement positif, demie période de sinusoide
    if z > H or z < 0:
        return 0
    else:
        return alpha*np.sin(z/H*np.pi)
        
def vent_sin(z, H, alpha): #sinusoide complète 
    if z > H or z < 0:
        return 0
    else:
        return alpha*np.sin(z/H*2*np.pi)
    
#Création d'une fonction vent general pouvant appliquer le profil de vent demandé par l'utilisateur
#Chaque profil de vent est associé à un numéro qui doit etre précisé au niveau du paramètre types_vent lors de l'appel de la fonction"""
def vent_general(z, altitudes, types_vent, alpha):
    H = altitudes[len(altitudes)-1]
    if z > H or z < 0:
        return 0
    else:
        j = 0
        while z > altitudes[j]:
            j+=1
        if types_vent[j] == 1: 
            return vent_constant(z, H, alpha)
        elif types_vent[j] == 2:
            return vent_lineaire(z, H, alpha)
        elif types_vent[j] == 3:
            return vent_sin_pos(z, H, alpha)
        elif types_vent[j] == 4:
            return vent_sin(z, H, alpha)

#Création des coefficients de diffusion        
def diffusion_constante(z, H, alpha):
    return alpha

def diffusion_sin_pos(z, H, alpha):
    return alpha*np.sin(z/H*np.pi)
    
        
#Création du nuage 1D
    #Deux paramètres : une épaisseur, un nombre de particules
def nuage_initial(epaisseur, nombre_particules):
    n = np.zeros(nombre_particules)   #n: position initiale de la particule, on créé un vecteur nul qui se remplira des positions de chaque particule dans la boucle for
    e = epaisseur / nombre_particules #e: distance entre chaque particule 
    i = 0
    for i in range(nombre_particules):
        n[i] = (i * e) + e
    return n

#Modélisation du transport du nuage avec profil de vent simpliste sans turbulence
    #Fonction pour résolution ED01 afin d'estimer la position de chaque particule en t+1
def euler_deterministe(nuage, pas, H, alpha, f): #nuage: nombre de particules
    n = np.zeros(len(nuage))
    for i in range(len(nuage)):
        n[i] = nuage[i] + pas * f(nuage[i], H, alpha)
    return n
       
#Introduction d'une turbulence aléatoire gaussienne dont l'efficacité est caractérisée par un coefficient de diffusion d
    #Création du générateur aléatoire gaussien       
def generateur(nb_point, mean, std):
    x = np.random.rand(nb_point)
    y = np.random.rand(nb_point)
    X = mean + std*(np.sqrt(-2*np.log(x)))*np.cos(2*np.pi*y) #formule démontrée dans le rapport
    return X

    #Création de la foncion de répartition      
def repartition(x):
    X,cn=np.unique(x,return_counts=True)
    Y=np.cumsum(cn)/np.sum(cn)
    plt.close()
    return X,Y

    #Définition test couplé KS-Shapiro, permet de tester la normalité du générateur aléatoire
def testKS(x):
    Xc=(x-np.mean(x))/np.std(x)
    Xt,Fn=repartition(Xc)
    F=scs.norm.cdf(Xt)
    Dn1=np.max(np.abs(F-Fn))
    Fng=np.append(0,Fn[0:-1])
    Dn2=np.max(np.abs(F-Fng))
    Dn=np.max([Dn1,Dn2])
    c=np.sqrt(x.shape)*Dn
    imax=int(np.sqrt(8*np.log(10)/(2*c**2)))
    somme=0
    i=1
    while i<imax+1:
        somme +=2*(-1)**(i-1)*np.exp(-2*(c*i)**2)
        i+=1
    if somme < 0.05 and shapiro(x)[1] < 0.05: #si les p-value des deux test inférieures à 5% alors le test est rejeté et la simulation est réinitialisée
        x = x*0
        return x
    else: 
        return x
    
    #Résolution ED01 avec introduction des turbulences aléatoires afin de déterminer la position de chaque particule en t+1
def euler_stoch(nuage, altitudes, types_vent, pas, H, alpha, d):
    n = np.zeros(len(nuage))
    g2 = generateur(len(nuage), 0, 2) 
    g = testKS(g2) #Il faut que les test de normalité soient vérifiés pour que Euler fonctionne
    for i in range(len(nuage)):
        n[i] = nuage[i] + pas * vent_general(nuage[i], altitudes, types_vent, alpha) + d(nuage[i], H, alpha)*np.sqrt(pas)*g[i]
        if n[i] < 0.:
            n[i] = 0.
    return n    

    #Etat d'un nuage à un instant T selon le profil de vent et les turbulences aléatoires  
def progression_stoch(nuage, altitudes, types_vent, pas, alpha, T, d):    
    m = int(T/pas)
    x = len(nuage)
    H = max(altitudes)
    matr = np.zeros((x, m))
    matr[:,0] = nuage
    ext_nuage = np.zeros((1, m))
    cent_gra = np.zeros((1, m))
    re = np.linspace(0,T,m)
    for i in range(1, int(T/pas)):
        matr[:,i] = euler_stoch(matr[:,i-1], altitudes, types_vent, pas, H, alpha, d)
        ext_nuage[:,i] = abs(max(matr[:,i]) - min(matr[:,i])) #permet de calculer l'extension du nuage
        cent_gra[:,i] = abs(np.mean(matr[:,i])) #permet de calculer le centre de gravité du nuage

#Représentations graphiques
    #Représentation de l'épaisseur du nuage en fonction du temps
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(re, ext_nuage[0, :], "blue")
    plt.title("Evolution de l'épaisseur du nuage en mètres (top) et de \nl'altitude de son centre de gravité (bottom)")
    plt.ylabel("Épaisseur (m)")

    #Représentation de la hauteur du centre de gravité du nuage en fonction du temps
    plt.subplot(2, 1, 2)
    plt.plot(re, cent_gra[0, :], "red")
    plt.xlabel("Temps (s)")
    plt.ylabel("Hauteur (m)")
    plt.savefig("../plots/Evolution de l'épaisseur du nuage en mètres")
    plt.show() 
    
    
    z = np.linspace(0,H*1.1,H*10)
    vent = []
    for i in range(len(z)): #Représentation des différents profils de vent appliqués selon les couches atmosphériques
        vent.append(vent_general(z[i], altitudes, types_vent, alpha))
    plt.figure(2)
    plt.plot(vent,z)
    plt.title("Profil de vent au cours du temps")
    plt.xlabel("Vitesse (m/s)")
    plt.ylabel("Hauteur (m)")
    plt.savefig("../plots/Profil de vent au cours du temps")
    plt.show()
    
    if len(nuage) > 10: #Lorsqu'il y a plus de 10 particules, l'histogramme ci dessous s'affiche
        B = matr[:,-1]
        Bc = np.linspace(np.min(B), np.max(B), np.sqrt(len(nuage)))
        h = np.histogram(B , bins = Bc)
        plt.figure(3)
        plt.bar(h[1][:-1], h[0], h[1][1:]-h[1][:-1], color = "b")
        plt.title("Dispersion de %g particules par rapport à la couche limite à %g mètres \naprès %g secondes" %(len(nuage), H, T))
        plt.xlabel("Altitute des particules (m)")
        plt.ylabel("Nombre de particules")
        plt.savefig("../plots/Dispersion de %g particules par rapport à la couche limite")
        plt.show()
    
    
    else:  #Représentations de l'évolution du nuage de particule au cours du temps
        plt.figure(3)
        for j in range(0, x):
            plt.plot(re, matr[j])
        plt.title("Evolution du nuage sur une hauteur de %g mètres \npendant une durée de %g secondes" %(H, T))
        plt.xlabel("Temps (s)")
        plt.ylabel("Hauteur (m)")
        plt.savefig("../plots/Evolution du nuage sur une hauteur de %g mètres \npendant une durée de %g secondes" %(H, T))
        plt.show()


# Création de l'interface interactive
def main():
    # Définition du types d'atmosphère
    a = int(input("Altitude de la couche limite ? "))
    b = int(input("Combien de couches souhaitez-vous ? "))
    altitudes = np.linspace(0,b-1,b)
    types_vent = np.linspace(0,b-1,b)
    test1 = -1
    test2 = -1
    alt = -2
    for i in range(b-1):
        print("Couche", i)
        while alt <= test1:
            print("Veuillez entrer des altitudes seuil croissantes")
            alt = int(input("Altitude ? " ))
        altitudes[i] = alt
        test1 = alt
        
        while test2 != 1 and test2 != 2 and test2 != 3 and test2 != 4:
            print("Type de vent ? : 1 = Vent Constant , 2 = Vent Linéraire , 3 = Vent sinusoïdal positif , 4 = Vent sinusoïdal")
            test2 = int(input("Vent ? "))
        types_vent[i] = test2
        test2 = -1
        
    altitudes[b-1] = a
    
    while test2 != 1 and test2 != 2 and test2 != 3 and test2 != 4:
            print("Type de vent ? : 1 = Vent Constant , 2 = Vent Linéraire , 3 = Vent sinusoïdal positif , 4 = Vent sinusoïdal")
            test2 = int(input("Vent ? "))        
    types_vent[b-1] = test2       
    
    print(altitudes)
    print(types_vent)
    alpha = float(input("Quel coefficient des vents souhaitez-vous ? "))
    
    #Définition du nuage initial
    particules = int(input("De combien de particules se compose le nuage ? "))
    epaisseur = int(input("Quelle est l'épaisseur initiale du nuage ? "))
    nuage = nuage_initial(epaisseur, particules) 
    
    #Visualisation de la progression du nuage
    pas = float(input("Pas ? "))
    duree = int(input("Sur quelle durée souhaitez-vous observer le nuage ? "))
    progression_stoch(nuage, altitudes, types_vent, pas, alpha, duree, diffusion_constante)

main()
