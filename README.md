# Análise de Logística - Previsão de Ineficiência em Entregas

## 📋 Descrição do Projeto

Este projeto desenvolve um modelo preditivo para identificar entregas ineficientes antes que elas ocorram. Utilizando dados históricos de ordens de serviço e estatísticas de desempenho, aplicamos técnicas de **modelagem estatística** e **machine learning** para prever a probabilidade de ineficiência em uma entrega.

### 🎯 Objetivo

Construir um modelo capaz de:
- **Prever ineficiência de entregas** com base em características do cliente, tipo de serviço e dados históricos
- **Reduzir falhas operacionais** identificando padrões de risco antes da execução
- **Otimizar alocação de recursos** focando em entregas de alto risco
- **Melhorar a experiência do cliente** prevenindo entregas problemáticas

### 📊 Metodologia

- **Análise Exploratória**: Compreender a distribuição de ineficiências e correlações
- **Modelagem Estatística**: Testes de hipótese, análise de variância e correlações
- **Machine Learning**: Regressão logística, árvores de decisão, random forest e outros algoritmos de classificação
- **Validação**: Validação cruzada, métricas de desempenho (precisão, recall, F1-score)

---

## 📁 Base de Dados - `base_chamados.csv`

A base de dados contém informações de ordens de serviço (OS) com as seguintes colunas:

| Coluna | Descrição |
|--------|-----------|
| `numero_os` | Identificador único da Ordem de Serviço |
| `ec_codcliente` | Código único do cliente/estabelecimento |
| `dia_referencia` | Data de referência da OS (YYYY-MM-DD) |
| `contratante` | Nome ou código do contratante |
| `operador` | Operador responsável pela execução |
| `grupo_servico` | Categoria do serviço (ex: INSTALAÇÃO, MANUTENÇÃO) |
| `forneceu_numero_logradouro` | Binária: cliente forneceu número do logradouro (0/1) |
| `forneceu_complemento` | Binária: cliente forneceu complemento do endereço (0/1) |
| `endereco_igual_cadastro` | Binária: endereço corresponde ao cadastro (0/1) |
| `forneceu_telefone` | Binária: cliente forneceu telefone (0/1) |
| `dois_telefones` | Binária: cliente forneceu dois telefones (0/1) |
| `taxa_ineficiencia_cliente` | Taxa histórica de ineficiência do cliente (0-1) |
| `ineficiencia_os_anterior` | Indicador de ineficiência em OS anterior do cliente |
| `ineficiencia` | **TARGET**: Indicador de ineficiência da OS atual (0=eficiente, 1=ineficiente) |
| `sistema_abertura` | Sistema que registrou a abertura da OS |

### 📈 Resumo da Base

- **Variáveis de Entrada**: Informações sobre cliente, serviço, dados de contato e histórico
- **Variável Alvo**: `ineficiencia` (binária - classificação)
- **Tipo de Problema**: Classificação Supervisionada (Previsão Binária)

---

## 🚀 Como Usar

```bash
# Instalar dependências
pip install pandas numpy scikit-learn matplotlib seaborn

# Executar análise
python analise_base.py
```

## 📝 Estrutura do Projeto

```
logistica/
├── README.md                    # Este arquivo
├── .gitignore                   # Arquivos ignorados (base_chamados.csv, .env)
├── analise_base.py              # Script principal de análise
├── base_chamados.csv            # Base de dados (⚠️ NÃO enviado para GitHub)
└── .env                         # Variáveis de ambiente (⚠️ NÃO enviado para GitHub)
```
