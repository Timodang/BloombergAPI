import numpy as np
import pandas as pd
from src.classes.utilitaire import Utils

class Metrics:

    DISCRET_LABEL = "discret"
    CONTINU_LABEL = "continu"

    # Ce projet concerne uniquement des données en daily
    MONTHLY_FACTOR = 21
    ANNUALIZATION_FACTOR = 252

    """
    Classe permettant de calculer un ensemble de métriques de performance et de risque pour un portefeuille
    """
    def __init__(self, ptf_nav:pd.DataFrame, method:str, frequency:str, benchmark:pd.DataFrame = None):
        """
        :param ptf_nav: DataFrame contenant la NAV du portefeuille à chaque date
        :param method: méthode de calcul des rendements
        :param frequency: fréquence des données pour le calcul des rendements
        :param benchmark: DataFrame contenant la NAV du benchmark du portefeuille
        """
        self.nav: pd.DataFrame = ptf_nav
        self.bench: pd.DataFrame = benchmark
        self.method: str = method
        self.frequency: str = frequency

        self.utils: Utils = Utils()

        self.annualization_factor: float = Metrics.ANNUALIZATION_FACTOR
        self.monthly_factor: float = Metrics.MONTHLY_FACTOR

    def compute_performance(self)->dict:
        """
        Méthode permettant de calculer le rendement annualisé et le rendement total d'une stratégie.
        :return: dictionnaire contenant le rendement annualisé et le rendement total
        """

        # Récupération des rendements
        ret_ptf: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)

        # Calcul du total return
        total_return: float = (self.nav.iloc[-1] / self.nav.iloc[0]) - 1

        # Calcul du rendement annualisé
        annualized_return: float = (1+total_return) ** (self.annualization_factor / ret_ptf.shape[0]) - 1
        return {"total_return": total_return, "annualized_return": annualized_return}

    def compute_vol(self)->dict:
        """
        Méthode permettant de calculer la volatilité quotidienne, mensuelle et annualisée
        d'un portefeuille.
        :return: Un dictionnaire contenant la volatilité quotidienne, mensuelle et annualisée
        """
        ptf_ret: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)
        vol: float = np.std(ptf_ret)
        monthly_vol: float = vol * self.monthly_factor
        annualized_vol: float = vol * self.annualization_factor
        return {"daily vol":vol, "monthly vol": monthly_vol, "annualized vol":annualized_vol}

    def compute_sharpe_ratio(self, rf:float = 0)->float:
        """
        Méthode permettant de calculer le Sharpe ratio d'un portefeuille
        :param rf: taux sans risque (0 par hypothèse)
        :return: sharpe ratio du portefeuille
        """
        ann_ret:float = self.compute_performance()["annualized_return"]
        ann_vol: float = self.compute_vol()["annualized vol"]
        sharpe:float = (ann_ret - rf)/ann_vol
        return sharpe

    def compute_downside_vol(self)->dict:
        """
        Méthode permettant de calculer la volatilité à la baisse quotidienne, mensuelle et
        annualisée du portefeuille
        :return: Dictionnaire contenant la volatilité à la baisse quotidienne, mensuelle et
        annualisée du portefeuille
        """

        # Etape 1 : calcul de la différence entre rendement et taux sans risuqe (= rendement) et
        # récupération des cas où la diff est négative
        ptf_ret: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)
        neg_ptf_ret: pd.DataFrame = ptf_ret[ptf_ret < 0]

        # Etape 2 : Calcul de la volatilité sur les rendements negatifs
        downside_vol: float = np.std(neg_ptf_ret)

        # Calcul de la volatilité à la baisse mensuelle et annualisée
        monthly_downside_vol: float = downside_vol * np.sqrt(self.monthly_factor)
        annualized_downside_vol: float = downside_vol * np.sqrt(self.ANNUALIZATION_FACTOR)

        # Etape 3 : Récupération de la downside vol annualisée
        return {"daily downside vol":downside_vol,
                "monthly downside vol": monthly_downside_vol,
                "annualized downside vol":annualized_downside_vol}

    def compute_sortino(self, rf:float = 0)->float:
        """
        Méthode permettant de calculer le ratio de Sortino du portefeuille
        :return: ratio de Sortino du portefeuille
        """
        ann_ret:float = self.compute_performance()["annualized_return"]
        downside_vol: float = self.compute_downside_vol()['annualized downside vol']
        if downside_vol == 0:
            raise Exception("Calcul impossible pour une volatilité à la baisse nulle")
        sortino:float = (ann_ret - rf)/downside_vol
        return sortino

    def compute_max_draw_down(self)->float:
        """
        Méthode permettant de calculer le max draw down
        d'un portefeuille
        :return: MaxDrawDown du portefeuille
        """

        # Etape 1 : initialisation du hwm et de la liste contenant les drawdowns
        list_drawdowns: list = []
        hwm:float = self.nav[0]

        # Etape 2 : boucle pour calculer le drawdown a chaque date
        for i in range(self.nav.shape[0]):
            # Mise du hwm si nouveaux plus haut
            if self.nav[i] > hwm:
                hwm = self.nav[i]

            # Calcul du drawdown
            drawdown: float = (self.nav[i] - hwm)/hwm
            list_drawdowns.append(drawdown)

        # Etape 3 : calcul du max drawdown
        return min(list_drawdowns)

    def compute_historical_var(self, quantile:float = 5)->float:
        """
        Méthode permettant de calculer la VaR historique du portefeuille
        pour un quantile donné
        :return: la VaR historique pour un quantile donné
        """
        ret: pd.DataFrame = self.utils.compute_asset_returns(self.nav, self.frequency, self.method)
        var: float = np.percentile(ret, quantile)
        return var

    def display_stats(self, nom_ptf:str)->pd.DataFrame:
        """
        Méthode permettant de calculer les statistiques descriptives du portefeuille
        :return: DataFrame contenant les statistiques descriptives du portefeuille
        """
        #
        dict_perf: dict = self.compute_performance()
        dict_vol: dict = self.compute_vol()
        dict_downside_vol: dict = self.compute_downside_vol()

        ann_ret: float = dict_perf["annualized_return"]
        vol: float = dict_vol["daily vol"]
        monthly_vol: float = dict_vol["monthly vol"]
        ann_vol: float = dict_vol["annualized vol"]
        tot_ret: float = dict_perf["total_return"]
        sharpe: float = self.compute_sharpe_ratio()
        downside_vol: float = dict_downside_vol["daily downside vol"]
        monthly_downside_vol: float = dict_downside_vol["monthly downside vol"]
        ann_downside_vol: float = dict_downside_vol["annualized downside vol"]
        sortino: float = self.compute_sortino()
        mdd:float = self.compute_max_draw_down()
        var: float = self.compute_historical_var()

        stats_dict: dict = {
            "Rendement annualisé en % ":round(ann_ret * 100, 5),
            "Total return en %":round(tot_ret * 100, 5),
            "Volatilité en %":round(vol * 100, 5),
            "Volatilité mensuelle en %":round(monthly_vol*100, 5),
            "Volatilité annualisée en %":round(ann_vol * 100, 5),
            "Sharpe ratio annualisé":round(sharpe,4),
            "Downside Vol en %":round(downside_vol * 100, 5),
            "Downside Vol mensuelle en %":round(monthly_downside_vol * 100, 5),
            "Downside Vol annualisée en %":round(ann_downside_vol * 100, 5),
            "Ratio de Sortino annualisé":round(sortino, 5),
            "MaxDrawDown":round(mdd,5),
            "VAR historique (en %)": round(var * 100, 5)
        }

        return pd.DataFrame.from_dict(stats_dict, orient='index', columns=[nom_ptf])
