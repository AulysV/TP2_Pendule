# -*- coding: utf-8 -*-
"""
@author: Y. Vadée Le Brun

DESCRIPTION

Ce code stocke la classe pendule.
"""

"""
BIBLIOTHEQUES
"""
# import de la bibliothèque numpy (gestion de matrices et routines mathématiques) en lui donnant le surnom np
import numpy as np

"""
CLASSE PENDULE
"""

class Pendule:
    # Le constructeur du pendule. Notez que les paramètres (sauf L) ont une valeur par défaut.
    # Cela permet de ne pas tout spécifier à chaque fois.
    def __init__(self, L, g=9.81, theta0=0, omega0=0, small_angle=False):
        # La longueur du fil du pendule
        self.L = L
        # La constante de gravitation
        self.g = g
        # L'angle initial du pendule
        self.theta0 = theta0
        # La vitesse angulaire initiale du pendule
        self.omega0 = omega0
        # Un booléen pour savoir si ce pendule est étudié dans l'approximation des petits angles
        self.small_angle = small_angle

    # Cette fonction correspond au G du polycopié. Elle renvoie la dérivée du vecteur A.
    def derA(self, t, A):
        theta, omega = A
        dtheta = omega
        if not self.small_angle:
            domega = -(self.g / self.L) * np.sin(theta)
        else:
            domega = -(self.g / self.L) * theta
        return np.array([dtheta, domega])

    def CI(self):
        return np.array([self.theta0, self.omega0])

    def A_math(self, t):
        # Un message d'erreur est envoyé si l'utilisateur demande la solution exacte
        # alors que l'approximation des petits angles n'est pas faite
        msg = "Le pendule ne peut donner de solution exacte sans l'approximation des petits angles"
        assert self.small_angle, msg

        g, L = self.g, self.L
        theta0, omega0 = self.theta0, self.omega0
        A = np.empty((np.shape(t)[0], 2))
        w0 = np.sqrt(g / L)
        theta = theta0 * np.cos(w0 * t)
        omega = -w0 * theta0 * np.sin(w0 * t)
        A[:, 0] = theta
        A[:, 1] = omega
        return A

    def Em(self, A):
        L, g = self.L, self.g
        theta, omega = A[0], A[1]
        if self.small_angle:
            Em = 0.5 * L**2 * omega**2
        else:
            Em = 0.5 * L**2 * omega**2 + L * g * (1 - np.cos(theta))
        return Em