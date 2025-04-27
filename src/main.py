from datetime import datetime
from src.classes.utilitaire import Utils

start_date: datetime = datetime(2007, 4, 30)
end_date: datetime = datetime(2024, 12,31)

# Récupération d'un calendrier contenant les jours ouvrés
calendar: list = Utils.get_calendar(start_date, end_date)

# Récupération des compositions par définition selon à chaque fin de mois (pour limiter les extract)
calendar_eom: list = Utils.get_last_date_month(calendar)
a=3