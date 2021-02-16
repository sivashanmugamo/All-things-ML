# This code block is to reformat the wdbc.data file to CSV (Comma Separated Values)

import pandas as pd

wdbc_location = "./Input/wdbc.data"

pd.set_option("display.max_colwidth", 100)
wdbc_data = pd.read_csv(wdbc_location, delimiter=",", encoding="utf-8", header=None)

wdbc_data.rename(columns={
    0: 'id',
    1: 'diagnosis',
    2: 'fld_1',
    3: 'fld_2',
    4: 'fld_3',
    5: 'fld_4',
    6: 'fld_5',
    7: 'fld_6',
    8: 'fld_7',
    9: 'fld_8',
    10: 'fld_9',
    11: 'fld_10',
    12: 'fld_11',
    13: 'fld_12',
    14: 'fld_13',
    15: 'fld_14',
    16: 'fld_15',
    17: 'fld_16',
    18: 'fld_17',
    19: 'fld_18',
    20: 'fld_19',
    21: 'fld_20',
    22: 'fld_21',
    23: 'fld_22',
    24: 'fld_23',
    25: 'fld_24',
    26: 'fld_25',
    27: 'fld_26',
    28: 'fld_27',
    29: 'fld_28',
    30: 'fld_29',
    31: 'fld_30'
    }, inplace=True)

print(wdbc_data)

put_location = "./wdbc_data.csv"

wdbc_data.to_csv(put_location, sep=",", encoding="utf-8", index=None)
