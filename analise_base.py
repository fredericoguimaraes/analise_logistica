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
    'forneceu_numero_logradouro',
    'forneceu_complemento', 
    'endereco_igual_cadastro',
    'forneceu_telefone',
    'dois_telefones',
    'grupo_servico',
    'ineficiencia_os_anterior',
    'contratante', 
    'operador',
    'sistema_abertura'
]

# serie temporal do percntual de ineficiencia 

base['dia_referencia'] = pd.to_datetime(pd.to_datetime(base['dia_referencia'], format='%Y-%m-%d'))
base['mes'] = base['dia_referencia'].apply(lambda x: x.replace(day=1))

agg = pd.crosstab(index= base['dia_referencia'],columns=base['ineficiencia'],values=base['ineficiencia'],aggfunc='count').sort_values(by='dia_referencia',ascending=True).reset_index()
agg = agg.fillna(0)
agg['%'] = agg[1]/ (agg[1] + agg[0])
agg['total'] = agg[1] + agg[0]

agg[1].sum() / agg['total'].sum() # taxa de ineficiencia da base 

agg_tratada = agg.loc[agg['%'] >= (agg['%'].quantile(0.25) - (agg['%'].quantile(0.75) - agg['%'].quantile(0.25)) * 1.5) ].reset_index(drop=True)
agg_tratada = agg.loc[agg['%'] <= (agg['%'].quantile(0.75) + (agg['%'].quantile(0.75) - agg['%'].quantile(0.25)) * 1.5) ].reset_index(drop=True)
mensal = pd.crosstab(index= base['mes'],columns=base['ineficiencia'],values=base['ineficiencia'],aggfunc='count',normalize=0).sort_values(by='mes',ascending=True).reset_index()


fig = plt.figure(figsize=(8, 10))
ax = fig.add_subplot(1,2,1)
gr1 = plt.plot(agg_tratada['dia_referencia'].values,agg_tratada['%'].values*100,linewidth=2, color='#00AEEF',label='ineficiencia')
plt.ylabel('%')
plt.xlabel('Meses')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.legend()
plt.title('Percentual de ineficiencia por dia')

ax = fig.add_subplot(1,2,2)
gr1 = plt.plot(mensal['mes'].values,mensal[1].values*100,linewidth=2, color='#00AEEF',label='ineficiencia')
plt.ylabel('%')
plt.xlabel('Meses')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.legend()
plt.title('Percentual de ineficiencia por mês')
plt.show()


"""
TEMOS EM MÉDIA 12K ORDENS DE SERVIÇO POR DIA E 797 ORDENS DE SERVIÇO INEFICIENTE POR DIA
com uma taxa de ineficiencia de 7.4% por dia 
utilizando uma distribuição binomial para estimativas
"""

# analise por cliente

base['ec_codcliente'].nunique() # 1.451.330 clientes
base['ec_codcliente'].groupby(base['ineficiencia']).nunique() # 132k clientes já tiveram alguma  entrega ineficiente | 9% dos clientes
agg_clientes = pd.crosstab(index= base['ec_codcliente'],columns=base['ineficiencia'],values=base['ineficiencia'],aggfunc='count',margins=True).reset_index()


# analise de variáveis 

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

base.loc[base['grupo_servico']!= 'TROCA']['ineficiencia'].value_counts(normalize=True)