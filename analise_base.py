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

base = base.loc[base.duplicated()==False].sort_values(by='numero_os',ascending=True)

base_1 = base.loc[base['ineficiencia'] == 1].reset_index(drop=True)
base_0 = base.loc[base['ineficiencia'] == 0].reset_index(drop=True)

base_amostras = base_0[[ 'operador', 'grupo_servico', 'forneceu_numero_logradouro', 'endereco_igual_cadastro', 'forneceu_telefone', 'dois_telefones',  'ineficiencia_os_anterior', 'sistema_abertura']].value_counts(normalize=True).reset_index()
base_amostras['qtde'] = round(base_amostras['proportion'] * base_1['numero_os'].count(),0)

base_0_ajustada = pd.DataFrame()
import random

for i in range(len(base_amostras)):
    aux = base_0.loc[(base_0['operador'] == base_amostras['operador'][i]) & (base_0['grupo_servico'] == base_amostras['grupo_servico'][i]) & (base_0['forneceu_numero_logradouro'] == base_amostras['forneceu_numero_logradouro'][i]) & (base_0['endereco_igual_cadastro'] == base_amostras['endereco_igual_cadastro'][i]) & (base_0['forneceu_telefone'] == base_amostras['forneceu_telefone'][i]) & (base_0['dois_telefones'] == base_amostras['dois_telefones'][i]) & (base_0['ineficiencia_os_anterior'] == base_amostras['ineficiencia_os_anterior'][i]) & (base_0['sistema_abertura'] == base_amostras['sistema_abertura'][i]) ]
    aux = aux.loc[aux.index.isin(random.sample(sorted(aux.index),int(base_amostras['qtde'][i]))) == True].reset_index(drop=True)
    base_0_ajustada = pd.concat([aux,base_0_ajustada])

base_final = pd.concat([base_0_ajustada,base_1]).reset_index(drop=True)

# OPERADOR
# diferenca relevante -> OPL5 | OPL6 | OPL7
pd.crosstab(index=base_final['operador'],columns=base_final['ineficiencia'],values=base_final['numero_os'],aggfunc='nunique',normalize=0,margins=True)

# grupo_servico
# diferenca relevante -> CAÇA POS
pd.crosstab(index=base_final['grupo_servico'],columns=base_final['ineficiencia'],values=base_final['numero_os'],aggfunc='nunique',normalize=0,margins=True)


# forneceu_numero_logradouro
# diferenca relevante -> não
pd.crosstab(index=base['forneceu_numero_logradouro'],columns=base['ineficiencia'],values=base['numero_os'],aggfunc='nunique',normalize=0,margins=True)


# forneceu_complemento
pd.crosstab(index=base_final['forneceu_complemento'],columns=base_final['ineficiencia'],values=base_final['numero_os'],aggfunc='nunique',normalize=0,margins=True)
t =pd.crosstab(index=base['forneceu_complemento'],columns=base['ineficiencia'],values=base['numero_os'],aggfunc='nunique')
s.chi2_contingency(t)


# endereco_igual_cadastro
# diferenca relevante -> não
pd.crosstab(index=base['endereco_igual_cadastro'],columns=base['ineficiencia'],values=base['numero_os'],aggfunc='nunique',normalize=0,margins=True)

# forneceu_telefone
# diferenca relevante -> não
pd.crosstab(index=base['forneceu_telefone'],columns=base['ineficiencia'],values=base['numero_os'],aggfunc='nunique',normalize=0,margins=True)


# dois_telefones
# diferenca relevante -> não
pd.crosstab(index=base['dois_telefones'],columns=base['ineficiencia'],values=base['numero_os'],aggfunc='nunique',normalize=0,margins=True)
t =pd.crosstab(index=base['dois_telefones'],columns=base['ineficiencia'],values=base['numero_os'],aggfunc='nunique')
s.chi2_contingency(t)


# ineficiencia_os_anterior
# diferenca relevante -> sim
pd.crosstab(index=base['ineficiencia_os_anterior'],columns=base['ineficiencia'],values=base['numero_os'],aggfunc='nunique',normalize=0,margins=True)
t =pd.crosstab(index=base['ineficiencia_os_anterior'],columns=base['ineficiencia'],values=base['numero_os'],aggfunc='nunique')
s.chi2_contingency(t)


# sistema_abertura
# diferenca relevante -> SISTEMAS_LOGISTICA
pd.crosstab(index=base['sistema_abertura'],columns=base['ineficiencia'],values=base['numero_os'],aggfunc='nunique',normalize=0,margins=True)
t =pd.crosstab(index=base['sistema_abertura'],columns=base['ineficiencia'],values=base['numero_os'],aggfunc='nunique')
s.chi2_contingency(t)


base_final['ineficiencia_os_anterior'] = np.where(base_final['ineficiencia_os_anterior'] =='-',-1,base_final['ineficiencia_os_anterior'])

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(base_final[['operador']].drop_duplicates())
enc.transform(base_final[['operador']]).toarray()
base_final[enc.get_feature_names_out()] = enc.transform(base_final[['operador']]).toarray()
base_final.drop(columns=['numero_os', 'ec_codcliente', 'dia_referencia', 'contratante', 'operador', 'grupo_servico','taxa_ineficiencia_cliente','sistema_abertura','forneceu_telefone']).describe()
X_train, X_test, y_train, y_test = train_test_split(
    base_final.drop(columns=['numero_os', 'ec_codcliente', 'dia_referencia', 'contratante', 'operador', 'grupo_servico','taxa_ineficiencia_cliente','sistema_abertura','dois_telefones']), base_final['ineficiencia'].values, test_size=0.2, random_state=42)

# 'operador', 'grupo_servico', 'sistema_abertura'
clf = LogisticRegression(random_state=0).fit(X_train, y_train)

recall_score(y_test, clf.predict(X_test))
precision_score(y_test, clf.predict(X_test))

cm = confusion_matrix(y_test, clf.predict(X_test))
print(cm)


# 2. Visualize the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.show()


