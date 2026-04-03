import pandas as pd 
import json 

class Dataframe_IMU:
    def __init__(self, fichier_csv: str) -> None:
        self._fichier_csv = fichier_csv
        self._df = pd.read_csv(self._fichier_csv)

    def apply(self):
        df_filtre = self._df[self._df["message_type"].str.strip() == "ATTITUDE"].copy()

        if df_filtre.empty:
            print("error")
            self._df = pd.DataFrame()
            return

        def extract_json(raw_msg):
            try:
                data_str = json.loads(raw_msg)
                data = json.loads(data_str)
                msg_data = data.get("message", {})

                return pd.Series({
                    "time_ms": msg_data.get("time_boot_ms"),
                    "roll": msg_data.get("roll"),
                    "pitch": msg_data.get("pitch"),
                    "yaw": msg_data.get("yaw"),
                    "rollspeed": msg_data.get("rollspeed"),
                    "pitchspeed": msg_data.get("pitchspeed"),
                    "yawspeed": msg_data.get("yawspeed"),
                })
            except Exception as e:
                print(f"Erreur lors de la lecture : {e}")
                return pd.Series(dtype='float64')

        df_final = df_filtre["raw_message"].apply(extract_json)
        self._df = df_final.dropna(how='all').reset_index(drop=True)

    def get_dataframe(self):
        return self._df