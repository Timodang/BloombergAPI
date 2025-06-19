from abc import ABC, abstractmethod
from enum import Enum

from scipy.optimize import minimize

import numpy as np
import pandas as pd
from typing import Union


class WeightingSchemeType(Enum):
    EQUAL_WEIGHT = "Equal Weight"
    SHARPE = "Sharpe Ratio"
    RANK = "Ranking"

class WeightingScheme(ABC):
    @abstractmethod
    def compute_weights(self, signals: Union[pd.Series, pd.DataFrame]):
        pass




class EquallyWeighting(WeightingScheme):
    def __init__(self, quantile: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantile: float = quantile

    def compute_weights(self, signals: pd.Series):
        """
        Méthode permettant de construire un portefeuille équipondéré Long Short
        Return : liste de poids associés à chaque actif du portefeuille
        """

        # Initialisation de la liste contenant les poids du portefeuille
        weights_ptf: list = []

        # Récupération du nombre de signaux d'achat et de vente
        nb_buy_signals: int = signals.loc[signals > 0].shape[0]
        nb_sell_signals: int = signals.loc[signals < 0].shape[0]

        # Calcul du poids à appliquer aux titres sur lesquels on souhaite prendre une position short / longue
        weight_long: float = 1 / nb_buy_signals
        weight_short: float = 1 / nb_sell_signals

        for i in range(len(signals)):
            # Cas où le signal est un signal d'achat
            if signals[i] > 0:
                weights_ptf.append(weight_long)

            # Cas où le signal est un signal de vente
            elif signals[i] < 0:
                weights_ptf.append(-weight_short)

            # Autre cas : pas de prise de position
            else:
                weights_ptf.append(0)

        # Vérification que la somme des poids du portefeuille soit nulle
        if np.round(np.sum(weights_ptf), 5) != 0:
            raise Exception("Le portefeuille n'est pas market neutral")
        return weights_ptf


class RankingWeightingSignals(WeightingScheme):
    def __init__(self, quantile: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantile: float = quantile

    def compute_weights(self, signals: pd.Series):
        """
        Méthode permettant de construire un portefeuille Long-Only en adoptant une méthodologie de ranking
        list_signals : liste contenant les valeurs des signaux renvoyés par la stratégie
        """

        # Création d'une liste pour stocker les poids des titres sur lesquels on prend une position
        weights_ptf: list = [0] * signals.shape[0]

        # Ranking des titres selon les signaux
        ranks: pd.Series = signals.rank(method="max", ascending=True).astype(float)

        # Calcul du nombre de titres à long/short selon le quantile
        nb_stocks: int = int(np.ceil(ranks.shape[0] * self.quantile))

        # Récupération des titres sur lesquels on prend une position d'achat
        top_ranks: pd.Series = ranks.nlargest(nb_stocks + 1)

        # Récupération des titres sur lesquels on prend une position à la vente
        bottom_ranks: pd.Series = ranks.nsmallest(nb_stocks + 1)

        # somme des rangs
        top_ranks_sum: pd.Series = top_ranks.sum()
        top_index_ranks = ranks.isin(top_ranks)

        bottom_ranks_sum: pd.Series = bottom_ranks.sum()
        bottom_index_ranks = ranks.isin(bottom_ranks)

        # boucle pour calculer le poids associé à chaque titre selon le rang
        for i in range(len(ranks)):
            if top_index_ranks[i]:
                weights_ptf[i] = ranks[i] / top_ranks_sum
            elif bottom_index_ranks[i]:
                weights_ptf[i] = -ranks[i] / bottom_ranks_sum

        # Vérification que la somme des poids du portefeuille soit nulle
        if np.round(np.sum(weights_ptf), 5) != 0:
            raise Exception("Le portefeuille n'est pas market neutral")
        return weights_ptf