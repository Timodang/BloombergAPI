import blpapi
import pandas as pd
import datetime as dt

DATE = blpapi.Name("date")
ERROR_INFO = blpapi.Name("errorInfo")
EVENT_TIME = blpapi.Name("EVENT_TIME")
FIELD_DATA = blpapi.Name("fieldData")
FIELD_EXCEPTIONS = blpapi.Name("fieldExceptions")
FIELD_ID = blpapi.Name("fieldId")
SECURITY = blpapi.Name("security")
SECURITY_DATA = blpapi.Name("securityData")


class BLP:
    """
    Classe contenant plusieeurs méthodes permettant d'établir une connection
    avec Bloomberg pour récupérer les données. Les méthodes implémentées sont :
    - BDP
    - BDH
    - BDS
    """

    # -----------------------------------------------------------------------------------------------------

    def __init__(self):
        """
            Improve this
            BLP object initialization
            Synchronus event handling

        """
        # Create Session object
        self.session = blpapi.Session()

        # Exit if can't start the Session
        if not self.session.start():
            print("Failed to start session.")
            return

        # Open & Get RefData Service or exit if impossible
        if not self.session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        self.session.openService('//BLP/refdata')
        self.refDataSvc = self.session.getService('//BLP/refdata')

        print('Session open')

    # -----------------------------------------------------------------------------------------------------

    def bdh(self, strSecurity, strFields, startdate, enddate, per='DAILY', perAdj='CALENDAR',
            days='NON_TRADING_WEEKDAYS', fill='PREVIOUS_VALUE', curr=None):
        """
            Summary:
                HistoricalDataRequest ;

                Gets historical data for a set of securities and fields

            Inputs:
                strSecurity: list of str : list of tickers
                strFields: list of str : list of fields, must be static fields (e.g. px_last instead of last_price)
                startdate: date
                enddate
                per: periodicitySelection; daily, monthly, quarterly, semiannually or annually
                perAdj: periodicityAdjustment: ACTUAL, CALENDAR, FISCAL
                curr: string, else default currency is used
                Days: nonTradingDayFillOption : NON_TRADING_WEEKDAYS*, ALL_CALENDAR_DAYS or ACTIVE_DAYS_ONLY
                fill: nonTradingDayFillMethod :  PREVIOUS_VALUE, NIL_VALUE

                Options can be selected these are outlined in “Reference Services and Schemas Guide.”

            Output:
                A list containing as many dataframes as requested fields
            # Partial response : 6
            # Response : 5

        """

        # -----------------------------------------------------------------------
        # Create request
        # -----------------------------------------------------------------------

        # Create request
        request = self.refDataSvc.createRequest('HistoricalDataRequest')

        # Put field and securities in list is single value is passed
        if type(strFields) == str:
            strFields = [strFields]

        if type(strSecurity) == str:
            strSecurity = [strSecurity]

        # Append list of securities
        for strF in strFields:
            request.append('fields', strF)

        for strS in strSecurity:
            request.append('securities', strS)

        # Set other parameters
        request.set('startDate', startdate.strftime('%Y%m%d'))
        request.set('endDate', enddate.strftime('%Y%m%d'))
        request.set('periodicitySelection', per)
        request.set('periodicityAdjustment', perAdj)
        request.set('nonTradingDayFillMethod', fill)
        request.set('nonTradingDayFillOption', days)

        if curr is not None:
            request.set('currency', curr)
        # -----------------------------------------------------------------------
        # Send request
        # -----------------------------------------------------------------------

        requestID = self.session.sendRequest(request)
        print("Sending request")

        # -----------------------------------------------------------------------
        # Receive request
        # -----------------------------------------------------------------------

        dict_Security_Fields = {}
        list_msg = []

        while True:
            event = self.session.nextEvent()

            # Ignores anything that's not partial or final
            if (event.eventType() != blpapi.event.Event.RESPONSE) & (
                    event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                continue

            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()
            list_msg.append(msg)

            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

                # -----------------------------------------------------------------------
        # Exploit data
        # -----------------------------------------------------------------------

        dict_output = {}
        # Create dictionary per field
        for field in strFields:
            # globals()[field] = {}
            dict_output[field] = {}
            for ticker in strSecurity:
                dict_output[field][ticker] = {}

        # Loop on all messages
        for msg in list_msg:
            countElements = 0
            ticker = msg.getElement(SECURITY_DATA).getElement(SECURITY).getValue()  # Ticker

            # dict_output[field][str(ticker)] = {}

            # numElements : permet de compter le nombre d'éléments qui sont renvoyés
            # Utilisation d'un indice ==> récupération de field.Name et field.getValue

            # Loop on dates
            for field_data in msg.getElement(SECURITY_DATA).getElement(FIELD_DATA):

                # Loop on different fields
                dat = field_data.getElement(0).getValue()

                for i in range(1, field_data.numElements()):
                    field = field_data.getElement(i)
                    dict_output[str(field.name())][ticker][dat] = field.getValue()

                countElements = countElements + 1 if field_data.numElements() > 1 else countElements

            if countElements == 0:
                for field in strFields:
                    del dict_output[field][ticker]

            # Remove ticker
        for field in dict_output:
            dict_output[field] = pd.DataFrame().from_dict(dict_output[field])

        return dict_output

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def bdp(self, strSecurity, strFields, strOverrideField='', strOverrideValue=''):

        """
            Summary:
                Reference Data Request ; Real-time if entitled, else delayed values
                Only supports 1 override


            Input:
                strSecurity
                strFields
                strOverrideField
                strOverrideValue

            Output:
               Dict
        """

        # -----------------------------------------------------------------------
        # Create request
        # -----------------------------------------------------------------------

        # Create request
        request = self.refDataSvc.createRequest('ReferenceDataRequest')

        # Put field and securities in list is single field passed
        if type(strFields) == str:
            strFields = [strFields]

        if type(strSecurity) == str:
            strSecurity = [strSecurity]

        # Append list of fields
        for strD in strFields:
            request.append('fields', strD)

        # Append list of securities
        for strS in strSecurity:
            request.append('securities', strS)

        # Add override
        if strOverrideField != '':
            o = request.getElement('overrides').appendElement()
            o.setElement('fieldId', strOverrideField)
            o.setElement('value', strOverrideValue)

        # -----------------------------------------------------------------------
        # Send request
        # -----------------------------------------------------------------------

        requestID = self.session.sendRequest(request)
        print("Sending request")

        # -----------------------------------------------------------------------
        # Receive request
        # -----------------------------------------------------------------------

        list_msg = []

        while True:
            event = self.session.nextEvent()

            # Ignores anything that's not partial or final
            if (event.eventType() != blpapi.event.Event.RESPONSE) & (
                    event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                continue

            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()

            list_msg.append(msg)

            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

                # -----------------------------------------------------------------------
        # Extract the data
        # -----------------------------------------------------------------------
        dict_output = {}

        for msg in list_msg:

            for secData in msg.getElement(SECURITY_DATA):
                ticker = secData.getElement(SECURITY).getValue()
                dict_output[str(ticker)] = {}

                for i in range(0, secData.getElement(FIELD_DATA).numElements()):
                    fieldData = secData.getElement(FIELD_DATA).getElement(i)
                    dict_output[ticker][str(fieldData.name())] = fieldData.getValue()

        return pd.DataFrame().from_dict(dict_output)

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def closeSession(self):
        print("Session closed")
        self.session.stop()

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def bds(self, strSecurity, strFields, strOverrideField='', strOverrideValue=''):

        """
            Summary:
                Reference Data Request ; Real-time if entitled, else delayed values
                Only supports 1 override


            Input:
                strSecurity
                strFields
                strOverrideField
                strOverrideValue

            Output:
               Dict
        """

        # -----------------------------------------------------------------------
        # Create request
        # -----------------------------------------------------------------------

        # Create request
        request = self.refDataSvc.createRequest('ReferenceDataRequest')

        # Put field and securities in list is single field passed
        if type(strFields) == str:
            strFields = [strFields]

        if type(strSecurity) == str:
            strSecurity = [strSecurity]

        # Append list of fields
        for strD in strFields:
            request.append('fields', strD)

        # Append list of securities
        for strS in strSecurity:
            request.append('securities', strS)

        # Add override
        if strOverrideField != '':
            o = request.getElement('overrides').appendElement()
            o.setElement('fieldId', strOverrideField)
            if type(strOverrideValue) is dt.datetime:
                o.setElement('value', strOverrideValue.strftime('%Y%m%d'))
            else:
                o.setElement('value', strOverrideValue)

        # -----------------------------------------------------------------------
        # Send request
        # -----------------------------------------------------------------------

        requestID = self.session.sendRequest(request)
        print("Sending request")

        # -----------------------------------------------------------------------
        # Receive request
        # -----------------------------------------------------------------------

        # Récupération de la requete
        list_msg = []
        dict_security_fields = {}

        while True:
            event = self.session.nextEvent()

            # Ignores anything that's not partial or final
            if (event.eventType() != blpapi.event.Event.RESPONSE) & (
                    event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                continue

            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()

            list_msg.append(msg)

            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

                # -----------------------------------------------------------------------
        # Extract the data
        # -----------------------------------------------------------------------

        dict_output = {}
        # Create dictionary per field
        for field in strFields:
            # globals()[field] = {}
            dict_output[field] = {}
            for ticker in strSecurity:
                dict_output[field][ticker] = {}

        # Boucle sur les messages (= boucle sur les tickers)
        for msg in list_msg:
            # Boucle sur les titres
            for sec in msg.getElement(SECURITY_DATA):
                ticker = sec.getElement(SECURITY).getValue()
                # Boucle sur les fields
                for field in strFields:
                    for field_data in sec.getElement(FIELD_DATA):
                        for sub_field_data in field_data:
                            sec_name = sub_field_data.getElement(0).getValue()

                            # Création d'un dictionnaire pour chaque titre de l'indice
                            dict_output[field][ticker][sec_name] = {}

                            # Boucle pour récupérer la valeur du field associé à chaque indice
                            for i in range(1, sub_field_data.numElements()):
                                field_name = str(sub_field_data.getElement(i).name())
                                dict_output[field][ticker][sec_name][field_name] = sub_field_data.getElement(
                                    i).getValue()

                # dict_output[field][ticker] = list_security
        return pd.DataFrame().from_dict(dict_output)

        # -----------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------