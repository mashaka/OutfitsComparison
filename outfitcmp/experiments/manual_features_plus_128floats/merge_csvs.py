import pandas as pd

CSV_1_PATH = 'data/test_m.csv'
CSV_2_PATH = 'data/out_128.csv'
CSV_RESULT_PATH = 'data/test.csv'


csv1 = pd.read_csv(CSV_1_PATH)
csv2 = pd.read_csv(CSV_2_PATH)

new = csv1.merge(csv2, 'left', 'image_name')
new.to_csv(CSV_RESULT_PATH,index=False)