# -*- coding: utf-8 -*-
"""
@author: Y. Vadée Le Brun

DESCRIPTION

Ce code est un simple test de fonctionnement.
On veut rédoudre les équations du mouvement d'un pendule plan et vérifier la cohérence de la solution.
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

# TODO