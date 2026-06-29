import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as s
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 320)
pd.options.display.float_format = '{:.2f}'.format


base = pd.read_csv('base_chamados.csv')


# base desbalanceada
base['ineficiencia'].value_counts(normalize=True)
base['dia_referencia'] = pd.to_datetime(base['dia_referencia'])
base['anomes'] = base['dia_referencia'].dt.to_period('M')
base['anomes'] = base['anomes'].dt.to_timestamp()

agg_dia = base['ineficiencia'].groupby(base['anomes']).sum().reset_index()


plt.plot(agg_dia['anomes'],agg_dia['ineficiencia'])
plt.show()