import pandas as pd
import pickle
from preprocessing import STD, fill_data, ORD, delete_cols, pipeline


def predict(session):
    filename = 'model/final_model.pickle'
    with open(filename, 'rb') as file:
        pipeline(session)
        final_session = session.drop(columns=['session_id'])
        rfc = pickle.load(file)
        return rfc.predict(final_session)


def localhost(num):
    value = str(predict(
        pd.read_csv('data/ga_sessions.csv', low_memory=False).iloc[
            num].to_frame().T))
    return value