# -*- coding: utf-8 -*-
"""
@author: Y. Vadée Le Brun

DESCRIPTION

Ce code propose plusieurs résolutions numériques du pendule plan.
Une quantification de l'erreur numérique est étudiée en fonction du pas de temps.
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
BIBLIOTHEQUES PERSONELLES
"""
# import des intégrateurs numériques présents dans integrateur_complet
from integrateur_complet import *
from pendule_plan import Pendule

"""
CODE PRINCIPAL
"""

""" INITIALISATION ET DEFINITION DES PARAMETRES DE SIMULATION"""
# longueur du fil du pendule en m
R = 0.5
# durée de la simulation (réglée sur 10 période)
t_max = 20
# Nombre de points voulu pour l'affichage
N = 101

# angle et vitesse angulaire initiale en rad
th_0 = np.pi/2
w_0 = 0

"""INITIALISATION VARIABLES DE CALCULS"""
# Liste des temps auquels sauvegarder les résultats pour les afficher
temps = np.linspace(0, t_max, N)
# Création du pendule
# On n'utilise pas l'approximation des petits angles de façoàn a affronter le problème réel
pendule = Pendule(L = R, theta0 = th_0, omega0 = w_0, small_angle = True)

# Création des listes de temps à tester pour chaque intégrateur
dt_list_euler = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6)
dt_list_midpoint = (0.1, 6e-2, 2e-2, 1e-2, 6e-3, 2e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
dt_list_RK4 = (0.2, 0.1, 6e-2, 2e-2, 1e-2, 6e-3, 2e-3, 1e-3, 6e-4, 2e-4, 1e-4, 6e-5, 2e-5)

# Dictionnaire des intégrateurs étudiés, de la couleur de tracé et du pas de temps utilisé pour chacun
solver_list = [{"solver_class": ForwardEuler, "color": "-r", "dt_list": dt_list_euler},
               {"solver_class": ExplicitMidpoint, "color": "-b", "dt_list": dt_list_midpoint},
               {"solver_class": RungeKutta4, "color": "-g", "dt_list": dt_list_RK4}]

# Comme la solution exacte n'est pas disponible, on calcule une solution de référence
# avec odeint en lui demandant une grande précision
start = time.perf_counter()
# J'ai gardé cette ligne pour quand je teste sans l'approximation des petits angles
A_ref = odeint(pendule.derA, pendule.CI(), temps, tfirst = True, rtol = 1e-12, atol = 1e-12)
A_ref = pendule.A_math(temps)
end = time.perf_counter()
elapsed = (end - start) * 1000
print("Le calcul de la solution de référence a duré", elapsed,"ms")

"""CALCULS"""
for item in solver_list:
    solver_class = item["solver_class"]
    dt_list = item["dt_list"]
    solver = solver_class(pendule)
    time_list, error_list = solver.return_error(temps, dt_list, A_ref)
    item["time_list"] = time_list
    item["error_list"] = error_list

"""SORTIE GRAPHIQUE"""
"""Figure pour l'erreur en fonction du pas de temps"""
plt.figure("Erreur en fonction du pas de temps")
for item in solver_list:
    plt.plot(item["dt_list"], item["error_list"], item["color"], lw=1.0, label=item["solver_class"].__name__)

plt.legend(loc='lower right')
plt.title("Erreur en fonction du pas de temps")
plt.xlabel("Pas de temps (s)")
plt.ylabel("Erreur (rad)")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)

"""Figure pour le temps d'exécution en fonction du pas de temps"""
plt.figure("Temps d'exécution en fonction du pas de temps")
for item in solver_list:
    plt.plot(item["dt_list"], item["time_list"], item["color"], lw=1.0, label=item["solver_class"].__name__)

plt.legend(loc='lower right')
plt.title("Temps d'exécution en fonction du pas de temps")
plt.xlabel("Pas de temps (s)")
plt.ylabel("Temps d'exécution (ms)")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)

plt.show()

