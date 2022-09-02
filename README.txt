Dataset: ACN-Dataset

    1. preprocessing

        Decimal encoding

 	        _id
 	        disconnectTime
 	        connectionTime
 	        doneChargingTime
            modifiedAt
 	        requestedDeparture

        One-Hot encoding

 	        paymentRequired (true: 1, false: 0)

    2. Features

        total_connection_time (TCT): disconnectTime - connectionTime
        done_charging_time_after_connection (DCTAC): doneChargingTime - connectioTime
        done_charging_time_before_disconnection (DCTBD): disconnectionTime - doneChargingTime
        kWhDelivered (KWD)
        WhPerMile (WPM)
        kWhRequested (KWR)
        milesRequested (MR)
        minutesAvailable (MA)
        modified_at_after_connection (MAAC): modifiedAt - connectionTime
        modified_at_before_disconnection (MABD): disconnectionTime - modifiedAt
        paymentRequired (PR)
        requestedDeparture_after_connection (RDAC): requestedDeparture - connectionTime