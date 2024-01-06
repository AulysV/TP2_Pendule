# -*- coding: utf-8 -*-
"""
@author: Y. Vadée Le Brun

DESCRIPTION

Ce code propose plusieurs résolutions numériques du pendule plan.
Une quantification de l'erreur numérique et une étude énergétique sont abordés.
"""

"""
BIBLIOTHEQUES
"""
# import de la bibliothèque numpy (gestion de matrices et routines mathématiques) en lui donnant le surnom np
import numpy as np
# import de la bibliothèque matplotlib (graphiques) en lui donnant le surnom plt
import matplotlib.pyplot as plt
# import de la bibliothèque time qui permet de mesurer le temps d'éxécution d'un programme
import time
# import du module integrate de la bibliothèque scipy qui dispose d'un integrateur de référence : odeint
from scipy.integrate import odeint

"""
BIBLIOTHEQUES PERSONNELLES
"""
# import des intégrateurs numériques présents dans integrateur_complet
from integrateur_complet import *
from integrateur_meca import *
from pendule_plan import Pendule

"""
FONCTIONS PERSONNELLES
"""
# Cette fonction fait les calculs exacts sur le pendule plan dans l'approximation des petits angles
# Elle renvoie la solution et l'énergie mécanique en fonction du temps
def calculs_pendule_approx(pendule):
    start = time.perf_counter()
    A_math = pendule.A_math(temps)
    end = time.perf_counter()
    elapsed = (end - start) * 1000
    print("Le calcul mathématique a duré", elapsed,"ms")
    em_math = pendule.Em(A_math.T)
    return A_math, em_math

# Cette fonction résoud numériquement un pendule avec différents intégrateurs possibles
# Pour chaque intégrateur, il faut préciser son numéro en partant de zéro, 
# sa couleur d'affichage et le pas de temps souhaité
# Pour chaque intégrateur, elle renvoie dans un tableau :
# La solution
# L'erreur par rapport à une solution de référence
# L'énergie mécanique en fonction du temps
def calculs_pendule(solver_list, pendule, A_ref, temps):
    for item in solver_list:
        solver_class = item["solver_class"]
        dt = item["dt"]
        # On mesure le temps d'exécution avec la bibliothèque time
        start = time.perf_counter()
        # Pour le calcul de la solution, la syntaxe d'odeint est différente de la notre
        # il faut donc prévoir une disjonction de cas pour si on l'utilise
        if solver_class == "Odeint":
            A = odeint(pendule.derA, pendule.CI(), temps, tfirst = True)
            solver_class_name = "odeint"
        else:
            solver = solver_class(pendule)
            A = solver.solve(pendule.CI(), temps, dt)
            solver_class_name = solver_class.__name__
        end = time.perf_counter()
        elapsed = (end - start) * 1000
        print("Le calcul via", solver_class_name, "a duré", elapsed,"ms")
        # Calcul de l'erreur à chaque pas de temps
        erreur = np.abs(A[:,0]-A_ref[:,0])
        # L'erreur maximale est le maximum de ce tableau
        erreur_max = max(erreur)
        print("L'erreur via", solver_class_name, "vaut", erreur_max)
        # Calcul de l'énergie mécanique
        em = pendule.Em(A.T)
        # On ajoute les trois grandeurs calculées au dictionnaire pour pouvoir y accéder plus tard
        item["A"] = A
        item["erreur"] = erreur
        item["em"] = em

def plot_A(solver_list):
    for item in solver_list:
        if item["solver_class"] == "Odeint": solver_name = "Odeint"
        else: solver_name = item["solver_class"].__name__
        plt.plot(temps, item["A"][:,0], item["color"], lw=1.0, label=solver_name)
    plt.legend(loc='lower right')
    plt.title("Angle de la balle au cours du temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)

def plot_erreur(solver_list):
    for item in solver_list:
        if item["solver_class"] == "Odeint": solver_name = "Odeint"
        else: solver_name = item["solver_class"].__name__
        plt.plot(temps, item["erreur"], item["color"], lw=1.0, label=solver_name)
    plt.legend(loc='lower right')
    plt.title("Erreur au cours du temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Erreur angulaire (rad)")
    plt.grid(True)

def plot_em(solver_list):
    for item in solver_list:
        if item["solver_class"] == "Odeint": solver_name = "Odeint"
        else: solver_name = item["solver_class"].__name__
        plt.plot(temps, item["em"], item["color"], lw=1.0, label=solver_name)
    plt.legend(loc='lower right')
    plt.title("Energie mécanique au cours du temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Energie mécanique (J/kg)")
    plt.grid(True)


"""
CODE PRINCIPAL
"""

""" INITIALISATION ET DEFINITION DES PARAMETRES DE SIMULATION"""
# longueur du fil du pendule en m
R = 0.5
# durée de la simulation (réglée sur 10 période)
t_max = 141.9
# Nombre de points voulu pour l'affichage
N = 10001

# angle et vitesse angulaire initiale en rad
th_0 = np.pi/2
w_0 = 0

# Dictionnaire des intégrateurs étudiés, de la couleur de tracé et du pas de temps utilisé pour chacun
#{"solver_class": ForwardEuler, "color": "-r", "dt": 1e-7} pas utilisé car trop lent ...
"""
solver_list = [{"solver_class": ExplicitMidpoint, "color": "-b", "dt": 6e-5},
               {"solver_class": RungeKutta4, "color": "-g", "dt": 9e-3},
               {"solver_class": VelocityVerlet, "color": "-c", "dt": 6e-5},
               {"solver_class": Stormer_Verlet, "color": "-y", "dt": 6e-5},
               {"solver_class": "Odeint", "color": "-m", "dt": 1e-3}  ]
               """

solver_list = [{"solver_class": ExplicitMidpoint, "color": "-b", "dt": 1e-4},
               {"solver_class": RungeKutta4, "color": "-g", "dt": 9e-3},
               {"solver_class": VelocityVerlet, "color": "-r", "dt": 4e-4}]

"""INITIALISATION VARIABLES DE CALCULS"""
# Liste des temps auquels sauvegarder les résultats pour les afficher
temps = np.linspace(0, t_max, N)
# Création du pendule
pendule_approx = Pendule(L = R, theta0 = th_0, omega0 = w_0, small_angle = True)
pendule_exact = Pendule(L = R, theta0 = th_0, omega0 = w_0, small_angle = False)         

"""ETUDE DANS LE CADRE DES PETITS ANGLES"""
"""Calculs"""
# Calcul de la solution exacte grace à la résolution mathématique :
A_math, em_math = calculs_pendule_approx(pendule_approx)

# Calcul du mouvement via les intégrateurs développés par nous
calculs_pendule(solver_list, pendule_approx, A_math, temps)

"""Figure pour l'angle en fonction du temps"""
plt.figure("Theta approx")
plt.plot(temps, A_math[:,0], "-k", lw=1.0, label="résolution analytique")
plot_A(solver_list)

"""Figure pour l'erreur en fonction du temps"""
plt.figure("Erreur approx")
plot_erreur(solver_list)

"""Figure pour l'énergie mécanique en fonction du temps"""
plt.figure("Energie approx")
plt.plot(temps, em_math, "-k", lw=1.0, label="résolution analytique")
plot_em(solver_list)

"""ETUDE EN DEHORS DES PETITS ANGLES"""
"""Calculs"""
# Comme la solution exacte n'est pas disponible, on calcule une solution de référence
# avec odeint en lui demandant une grande précision
start = time.perf_counter()
A_ref = odeint(pendule_exact.derA, pendule_exact.CI(), temps, tfirst = True, rtol = 1e-12, atol = 1e-12)
end = time.perf_counter()
elapsed = (end - start) * 1000
print("Le calcul de la solution de référence a duré", elapsed,"ms")
em_ref = pendule_exact.Em(A_ref.T)
# Calcul du mouvement via les intégrateurs développés par nous
calculs_pendule(solver_list, pendule_exact, A_ref, temps)

"""Figure pour l'angle en fonction du temps"""
plt.figure("Theta exact")
plt.plot(temps, A_ref[:,0], "-k", lw=1.0, label="résolution précise")
plot_A(solver_list)

"""Figure pour l'erreur en fonction du temps"""
plt.figure("Erreur exact")
plot_erreur(solver_list)

"""Figure pour l'énergie mécanique en fonction du temps"""
plt.figure("Energie exact")
plt.plot(temps, em_ref, "-k", lw=1.0, label="résolution précise")
plot_em(solver_list)

plt.show()