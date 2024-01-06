# -*- coding: utf-8 -*-
"""
DESCRIPTION

Ce code est un simple test de fonctionnement.
On veut rédoudre les équations du mouvement d'un pendule plan et vérifier la cohérence de la solution.
"""

"""
BIBLIOTHEQUES
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint

"""
BIBLIOTHEQUES PERSONNELLES
"""
from integrateur_complet import *
from pendule_plan import Pendule

"""
FONCTIONS PERSONNELLES
"""
def calculs_pendule_approx(pendule):
    start = time.perf_counter()
    A_math = pendule.A_math(temps)
    end = time.perf_counter()
    elapsed = (end - start) * 1000
    print("Le calcul mathématique a duré", elapsed, "ms")
    em_math = pendule.Em(A_math.T)
    return A_math, em_math

def calculs_pendule(solver_list, pendule, A_ref, temps):
    for item in solver_list:
        solver_class = item["solver_class"]
        dt = item["dt"]
        start = time.perf_counter()
        if solver_class == "Odeint":
            A = odeint(pendule.derA, pendule.CI(), temps, tfirst=True)
            solver_class_name = "odeint"
        else:
            solver = solver_class(pendule)
            A = solver.solve(pendule.CI(), temps, dt)
            solver_class_name = solver_class.__name__
        end = time.perf_counter()
        elapsed = (end - start) * 1000
        print("Le calcul via", solver_class_name, "a duré", elapsed, "ms")
        erreur = np.abs(A[:, 0] - A_ref[:, 0])
        erreur_max = max(erreur)
        print("L'erreur via", solver_class_name, "vaut", erreur_max)
        em = pendule.Em(A.T)
        item["A"] = A
        item["erreur"] = erreur
        item["em"] = em

def plot_A(solver_list):
    for item in solver_list:
        if item["solver_class"] == "Odeint":
            solver_name = "Odeint"
        else:
            solver_name = item["solver_class"].__name__
        plt.plot(temps, item["A"][:, 0], item["color"], lw=1.0, label=solver_name)
    plt.legend(loc='lower right')
    plt.title("Angle de la balle au cours du temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)

def plot_erreur(solver_list):
    for item in solver_list:
        if item["solver_class"] == "Odeint":
            solver_name = "Odeint"
        else:
            solver_name = item["solver_class"].__name__
        plt.plot(temps, item["erreur"], item["color"], lw=1.0, label=solver_name)
    plt.legend(loc='lower right')
    plt.title("Erreur au cours du temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Erreur angulaire (rad)")
    plt.grid(True)

def plot_em(solver_list):
    for item in solver_list:
        if item["solver_class"] == "Odeint":
            solver_name = "Odeint"
        else:
            solver_name = item["solver_class"].__name__
        plt.plot(temps, item["em"], item["color"], lw=1.0, label=solver_name)
    plt.legend(loc='lower right')
    plt.title("Energie mécanique au cours du temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Energie mécanique (J/kg)")
    plt.grid(True)

"""
CODE PRINCIPAL
"""

# Définir les paramètres de simulation
R = 0.5
t_max = 10
N = 1001
th_0 = np.pi / 2
w_0 = 0

# Initialiser les variables de calculs
temps = np.linspace(0, t_max, N)
pendule = Pendule(L=R, theta0=th_0, omega0=w_0, small_angle=True)

# Créer des listes de temps à tester pour chaque intégrateur
dt_list_euler = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6)
dt_list_midpoint = (0.1, 6e-2, 2e-2, 1e-2, 6e-3, 2e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
dt_list_RK4 = (0.2, 0.1, 6e-2, 2e-2, 1e-2, 6e-3, 2e-3, 1e-3, 6e-4, 2e-4, 1e-4, 6e-5, 2e-5)

# Dictionnaire des intégrateurs étudiés, de la couleur de tracé et du pas de temps utilisé pour chacun
solver_list = [{"solver_class": ForwardEuler, "color": "-r", "dt_list": dt_list_euler},
               {"solver_class": ExplicitMidpoint, "color": "-b", "dt_list": dt_list_midpoint},
               {"solver_class": RungeKutta4, "color": "-g", "dt_list": dt_list_RK4},
               {"solver_class": "Odeint", "color": "-m", "dt_list": (1e-3,)}]

# Comme la solution exacte n'est pas disponible, on calcule une solution de référence avec odeint
A_ref = odeint(pendule.derA, pendule.CI(), temps, tfirst=True, rtol=1e-12, atol=1e-12)

# Calculs pour chaque intégrateur
calculs_pendule(solver_list, pendule, A_ref, temps)

# Figures
plt.figure("Theta approx")
plt.plot(temps, A_ref[:, 0], "-k", lw=1.0, label="résolution analytique")
plot_A(solver_list)

plt.figure("Erreur approx")
plot_erreur(solver_list)

plt.figure("Energie approx")
plt.plot(temps, pendule.Em(A_ref.T), "-k", lw=1.0, label="résolution analytique")
plot_em(solver_list)

plt.show()
