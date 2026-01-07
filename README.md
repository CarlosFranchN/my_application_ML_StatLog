# 🏦 Análise de Risco de Crédito (Credit Scoring) - German Credit Data

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Concluído-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📄 Descrição do Projeto

Este projeto consiste no desenvolvimento de um modelo de Machine Learning para prever o risco de inadimplência (default) de clientes bancários. O objetivo não é apenas classificar corretamente, mas **maximizar o lucro da instituição financeira** através de uma **Matriz de Custos** personalizada e garantir a transparência das decisões via **SHAP (Explainable AI)**.

O dataset utilizado foi o clássico **Statlog (German Credit Data)**, que contém dados demográficos e histórico financeiro de 1.000 clientes.

---

## 💼 O Problema de Negócio

No cenário de crédito, os erros do modelo têm pesos financeiros diferentes:
* **Erro Tipo I (Falso Positivo):** Negar crédito a um bom pagador. O banco perde os juros (Custo de Oportunidade).
* **Erro Tipo II (Falso Negativo):** Dar crédito a um mau pagador. O banco perde o valor principal emprestado (Prejuízo Real).

**Premissa adotada neste projeto:** O prejuízo de um calote é **5x maior** que o lucro perdido de um bom cliente. O modelo foi otimizado para respeitar essa regra de negócio.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, Numpy
* **Visualização:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Random Forest / XGBoost - *ajuste conforme seu modelo*)
* **Explainability:** SHAP (SHapley Additive exPlanations)
* **Balanceamento:** SMOTE (Synthetic Minority Over-sampling Technique)

---

## 📊 Pipeline do Projeto

1.  **Análise Exploratória (EDA):** Identificação de padrões em variáveis como `Duration of Credit`, `Credit Amount` e `Checking Account Status`.
2.  **Pré-processamento:**
    * Codificação de variáveis categóricas (OneHotEncoding/LabelEncoder).
    * Tratamento de dados desbalanceados (70% bons / 30% ruins).
3.  **Modelagem:** Treinamento de algoritmos focados em otimização de *Recall* (para capturar os maus pagadores).
4.  **Avaliação Financeira:** Aplicação da Matriz de Custos para calcular o lucro estimado do modelo versus um cenário sem modelo.
5.  **Explicabilidade:** Uso do SHAP para entender quais features impactam a decisão de crédito.

---

## 📈 Principais Resultados

### 1. Performance do Modelo
* **Acurácia:** XX% (*Insira seu valor*)
* **Recall (Classe de Risco):** XX% (*Insira seu valor - métrica crucial*)
* **ROC-AUC:** 0.XX

### 2. Impacto Financeiro (Matriz de Custos)
Ao simular uma carteira de empréstimos, o modelo apresentou os seguintes resultados:

| Cenário | Resultado Financeiro |
| :--- | :--- |
| Sem Modelo (Aceitar Todos) | R$ -XXX.XXX (Prejuízo) |
| **Com Nosso Modelo** | **R$ +XXX.XXX (Lucro)** |

### 3. Fatores de Decisão (SHAP)
As variáveis que mais influenciaram o risco de crédito foram:
1.  Status da Conta Corrente (*Checking Account*)
2.  Duração do Empréstimo (*Duration*)
3.  Histórico de Crédito (*Credit History*)

*(Recomendo colocar aqui uma imagem do gráfico `shap.summary_plot`)*

---

## 🚀 Como Executar o Projeto

1. Clone o repositório:
```bash
git clone [https://github.com/SEU_USUARIO/NOME_DO_REPO.git](https://github.com/SEU_USUARIO/NOME_DO_REPO.git)
```

2. Instale as dependências:
```bash
    pip install -r requirements.txt
```


## Autor
[Carlos Franch Aragão]



### O que você precisa fazer agora:

1.  **Preencher os "XX%"** na seção de Resultados com os números reais do seu notebook.
2.  **Salvar as imagens:** Salve o gráfico do SHAP (`summary_plot`) e a Matriz de Confusão como imagens (png) numa pasta e coloque no README (posso te ensinar a linkar a imagem se precisar).
3.  **Requirements:** Lembre-se de gerar o `requirements.txt` (`pip freeze > requirements.txt`).

Ficou do jeito que você queria? Se quiser ajustar o tom para ser mais "técnico" ou mais "executivo", me avise!