from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib
import os


def delete_cols(df):
    if 'utm_keyword' in list(df.columns):
        del df['utm_keyword']
        del df['device_os']
        del df['device_model']
        del df['visit_date']
        del df['visit_time']
        del df['visit_number']
        del df['client_id']
    if df.shape[0] > 200000:
        for col in df.columns:
            if float(df[col].isnull().sum() / df.shape[0]) >= 0.2:
                del df[col]


def fill_data(df):
    obj_col = []
    num_col = []
    for col in df.columns:
        if df[col].dtype == 'object':
            obj_col.append(col)
        else:
            num_col.append(col)

    path = 'fill_na/'
    directory = os.listdir(path)
    if len(directory) < 3:
        for null_col in num_col:
            df[null_col].fillna(int(df[null_col].mean()), inplace=True)
            filename = f'{path}{null_col}_fd.txt'
            filled_data = int(df[null_col].mean())
            with open(filename, 'w') as f:
                f.write(str(filled_data))

        for null_col in obj_col:
            df[null_col].fillna(df[null_col].value_counts().idxmax(), inplace=True)
            filename = f'{path}{null_col}_fd.txt'
            filled_data = df[null_col].value_counts().idxmax()
            with open(filename, 'w') as f:
                f.write(filled_data)
    else:
        cols = [col for col in df.columns]
        for col in cols:
            filename = f'{path}{col}_fd.txt'
            with open(filename) as f:
                filled_data = f.read()
                if df[col].dtype == 'object':
                    df[col].fillna(filled_data, inplace=True)
                else:
                    df[col].fillna((filled_data), inplace=True)


def ORD(df):
    obj_col = [col for col in df.columns if df[col].dtype == 'object']
    obj_col.pop(0)
    path = 'encoders/'
    directory = os.listdir(path)
    if len(directory) < 3:
        for obj in obj_col:
            enc = LabelEncoder()
            enc.fit(df[[obj]])
            df[obj] = enc.transform(df[[obj]])
            filenam = f'{path}{obj}_encoder.bin'
            joblib.dump(enc, filenam)
    else:
        for obj in obj_col:
            enc = joblib.load(f'{path}{obj}_encoder.bin')
            lol = True
            df[obj] = enc.transform(df[[obj]])


# Standartization
def STD(df):
    path = 'scalers/'
    directory = os.listdir(path)
    num_col = [col for col in df.columns if df[col].dtype != 'object']
    if len(directory) == 0:
        for std_feature in num_col:
            std_scaler = StandardScaler()
            std_scaler.fit(df[[std_feature]])
            scaled = std_scaler.transform(df[[std_feature]])
            filenam = f'scalers/{std_feature}_scaler.bin'
            joblib.dump(std_scaler, filenam)
            df[f"{std_feature}"] = scaled
    else:
        for process in num_col:
            scaler = joblib.load(f'scalers/{process}_scaler.bin')
            scaled = scaler.transform(df[[process]])
            df[f"{process}"] = scaled





def pipeline(session):
    delete_cols(session)
    fill_data(session)
    ORD(session)
    STD(session)
