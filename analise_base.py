import pandas as pd 
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import chi2_contingency
import numpy as np
matplotlib.use('TkAgg')

desired_width = 320
pd.set_option('display.width', desired_width)
# np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns', 28)  # tela console

base = pd.read_csv('base_chamados.csv', sep=',', encoding='utf-8')
base.describe()

categorical_vars = [
    'contratante', 
    'operador',
    'sistema_abertura',
    'grupo_servico',
    'forneceu_numero_logradouro',
    'forneceu_complemento', 
    'endereco_igual_cadastro',
    'forneceu_telefone',
    'dois_telefones',
    'ineficiencia_os_anterior'

]

# 1. ANALISE GERAL BASE
base['dia_referencia'] = pd.to_datetime(pd.to_datetime(base['dia_referencia'], format='%Y-%m-%d'))
base['mes'] = base['dia_referencia'].apply(lambda x: x.replace(day=1))

# ineficiencia geral
base['ineficiencia'].value_counts(normalize=True) 

# --- ineficiencia por mes
agg_mes = pd.crosstab(index=base['mes'],columns=base['ineficiencia'],values=base['numero_os'],aggfunc='nunique',margins=True).reset_index()
agg_mes['%'] = (agg_mes[1]/agg_mes['All'])*100
agg_mes = agg_mes[:-1]
agg_mes[['All','%']].corr()

fig = plt.figure(figsize=(8, 10))
ax = fig.add_subplot()
gr1 = plt.plot(agg1['DATA_REF'].values,agg1['agropecuario'].values,linewidth=2, color='#00AEEF',label='Agro')


plt.ylabel('%')
plt.xlabel('Meses')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.legend()
plt.title('Percentual de Republicação em 1 hora')
plt.show()


""" 
--> ANOTACOES
* a base conta com mais de 2M de registros e cerca de 170k de registos com ineficiencia; Aproximadamente 7.4% de ineficiencia geral
* o aumento de registros está diretamente ligado ao aumento de ineficiena - Onde está concetrado o maior aumento de registros e por consequencia de ineficiencia? - 

""" 

# ANALISE DAS VARIAVEIS

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

results = []
for var in categorical_vars:
    contingency_table = pd.crosstab(base[var], base['ineficiencia'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    cramers = cramers_v(base[var], base['ineficiencia'])
    
    results.append({
        'Variável': var,
        'Chi2': chi2,
        'P-valor': p_value,
        'V Cramér': cramers,
        'Significante': 'Sim' if p_value < 0.05 else 'Não'
    })

results_df = pd.DataFrame(results).sort_values('V Cramér', ascending=False)
print(results_df)

pd.crosstab(index=base['mes'],columns=[base['grupo_servico'],base['ineficiencia']],values=base['numero_os'],aggfunc='nunique',margins=True)


# INFERENCIA 

base.loc[base['grupo_servico']!= 'TROCA']['ineficiencia'].value_counts(normalize=True)