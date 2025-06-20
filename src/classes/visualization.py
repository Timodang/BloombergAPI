import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Visualisation:
    """
    Cette classe sert exclusivement à centraliser les éléments d'affichages graphiques.

    Méthode à implémenter :
    - Track Record (performance cumulée)
    - Mesure de risque en rolling
    - Poids ?
    """

    def __init__(self):
        pass

    @staticmethod
    def plot_cumulative_returns(returns: pd.DataFrame,
                                title: str = "Rendements cumulés") -> None:
        """
        Calcule et affiche les rendements cumulés pour chaque actif.
        """
        cumulative_returns: pd.DataFrame = (1 + returns).cumprod()
        cumulative_returns.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Valeur")
        plt.legend(title="Actifs", frameon=True, fontsize=8)
        plt.axhline(y=1, color="red", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_exposure(exposition: pd.DataFrame,
                      title: str = "Exposition brute du portefeuille") -> plt.Axes:
        """
        Affiche l'exposition brute du portefeuille au cours du temps.

        :param exposition: DataFrame ou Series de l'exposition brute
        :param title: Titre du graphique
        :return: L'objet Axes
        """
        # Vérifier que les données existent
        if exposition is None or (hasattr(exposition, 'empty') and exposition.empty):
            raise ValueError("Les données d'exposition ne sont pas disponibles. Exécutez d'abord run_backtest().")

        # Si c'est un DataFrame à colonne unique, extraire la Series sous-jacente
        if isinstance(exposition, pd.DataFrame) and exposition.shape[1] == 1:
            series = exposition.iloc[:, 0]
        else:
            series = exposition.squeeze()

        # Trace de l'exposition brute au fil du temps
        fig, ax = plt.subplots(figsize=(12, 6))
        series.plot(ax=ax, legend=False)

        # Titres et labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Exposition brute", fontsize=12)

        # Formater l'axe des ordonnées en pourcentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # Grille légère
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        return ax