from datetime import datetime
import pandas as pd
import numpy as np
from src.classes.utilitaire import Utils

# Pip install pour blpapi

test = np.nan/np.nan

start_date: datetime = datetime(1998, 2, 2)
end_date: datetime = datetime(2024, 12,31)

# Récupération d'un calendrier contenant les jours ouvrés
calendar: list = Utils.get_calendar(start_date, end_date)

# Récupération des compositions par définition selon à chaque fin de mois (pour limiter les extract)
calendar_eom: list = Utils.get_last_date_month(calendar)
dates: pd.DataFrame = pd.DataFrame(calendar)

# Récupération des dates de fin de mois
dates_eom: pd.DataFrame = pd.DataFrame(calendar_eom)

# Récupération des dates par semaine
calendar_eow: list = Utils.get_last_date_week(calendar)
dates_eow: pd.DataFrame = pd.DataFrame(calendar_eow)

"""
Première étape : Importation des données
"""

"""
Deuxième étape : Réalisation du backtest
"""

"""
Troisième étape : Etude des performances
"""

"""
Quatrième étape : Export pour Bloomberg
"""