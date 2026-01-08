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

### 1. Performance dos Modelos

## 📊 Comparativo de Modelos

Como o objetivo do negócio é evitar o calote, a métrica mais importante para nós é o **Recall da Classe 2 (Maus Pagadores)**. Um modelo com alta acurácia que não detecta os caloteiros não serve para o banco.

| Modelo | Acurácia Global | Recall (Maus Pagadores) | F1-Score (Maus Pagadores) | Observação |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 66% | **0.60** 🏆 | 0.50 | Melhor detector de risco |
| **Random Forest** | 73% | 0.51 | **0.52** | Melhor equilíbrio geral |
| Support Vector Machine | **75%** | 0.38 | 0.47 | Alta acurácia, mas deixa passar muitos riscos |
| Decision Tree | 67% | 0.32 | 0.35 | Baixa performance em risco |
| K-Nearest Neighbors | 70% | 0.05 ❌ | 0.08 | Incapaz de detectar caloteiros |

> **Conclusão:** Apesar do SVM ter a maior acurácia (75%), a **Regressão Logística** se mostrou mais viável para o negócio por identificar 60% dos maus pagadores, contra apenas 38% do SVM.


### 2. Impacto Financeiro (Matriz de Custos)
Para demonstrar a aplicabilidade prática dos modelos, simulamos uma operação de crédito real. Em problemas de risco, a métrica técnica (Acurácia) é menos importante do que o **Lucro Líquido**.

#### 1. Premissas da Simulação
Adotamos os seguintes valores para cada cliente da base de teste:
* **Empréstimo Médio:** R$ 5.000,00
* **Lucro (Juros Recebidos):** R$ 2.000,00 (Para bons pagadores aprovados)
* **Prejuízo (Inadimplência):** -R$ 5.000,00 (Para maus pagadores aprovados)

#### 2. Resultados da Simulação

Comparativo do Lucro Líquido gerado por cada modelo versus um cenário base (sem modelo de IA).

| Modelo | Resultado Financeiro | Lucro Extra vs. Sem Modelo | Performance |
| :--- | :--- | :--- | :---: |
| **Logistic Regression** | **R$ 121.000,00** | **+ R$ 112.000,00** | 🏆 **Campeão** |
| Random Forest | R$ 118.000,00 | + R$ 109.000,00 | 🥈 Vice |
| Decision Tree | R$ 112.000,00 | + R$ 103.000,00 | 🥉 3º Lugar |
| *Sem Modelo (Baseline)* | *R$ 9.000,00* | *R$ 0,00* | ⚠️ Risco Alto |
| K-Nearest Neighbors | R$ 1.000,00 | <span style="color:red">- R$ 8.000,00</span> | ❌ Prejuízo |


#### 3. Conclusão de Negócio

* **A Melhor Escolha:** A **Regressão Logística** foi o modelo mais eficiente. Apesar de ter uma acurácia global menor que o SVM ou Random Forest, ela teve o melhor desempenho na detecção de caloteiros (Recall da Classe 2), maximizando o lucro final.
* **O Perigo do KNN:** O modelo KNN apresentou um desempenho financeiro **pior do que não ter modelo nenhum** (R$ 1.000 vs R$ 9.000 do baseline). Isso ocorre porque ele falhou em identificar os perfis de risco, aprovando empréstimos que resultaram em prejuízo massivo.


### 4. Diagnóstico Inicial: O Risco do Limiar Padrão (0.5)

Inicialmente, rodamos a simulação financeira utilizando o **threshold padrão de 50%** (ou seja, só negamos crédito se a certeza de calote for > 0.5).

**O resultado foi desastroso:** Como o custo do calote é muito alto (5x o lucro), ser "leniente" gerou prejuízo em **todos os modelos**. Isso prova que usar Machine Learning "fora da caixa" sem alinhamento com o negócio é perigoso.

<details>
  <summary>🔻 Clique para expandir os Logs de Simulação (Cenário de Prejuízo)</summary>

```text
--- Simulação Financeira (Decision Tree) ---
Clientes Bons Aprovados: 126 (Lucro: R$ 126.000)
Calotes Tomados: 28 (Prejuízo: R$ -140.000)
RESULTADO LÍQUIDO: R$ -55.000,00 (PREJUÍZO)

--- Simulação Financeira (Random Forest) ---
Clientes Bons Aprovados: 139 (Lucro: R$ 139.000)
Calotes Tomados: 32 (Prejuízo: R$ -160.000)
RESULTADO LÍQUIDO: R$ -49.000,00 (PREJUÍZO)

--- Simulação Financeira (Logistic Regression) ---
Clientes Bons Aprovados: 118 (Lucro: R$ 118.000)
Calotes Tomados: 23 (Prejuízo: R$ -115.000)
RESULTADO LÍQUIDO: R$ -46.000,00 (PREJUÍZO)

--- Simulação Financeira (K-Nearest Neighbors) ---
Clientes Bons Aprovados: 83 (Lucro: R$ 83.000)
Calotes Tomados: 33 (Prejuízo: R$ -165.000)
RESULTADO LÍQUIDO: R$ -166.000,00 (PREJUÍZO CRÍTICO)
```
</details>

#### 5. A Solução: Otimização do Limiar (Threshold Tuning)

Visto que o padrão gerou prejuízo, realizamos uma análise de sensibilidade variando a régua de corte. O objetivo foi encontrar o "Sweet Spot": o ponto exato onde maximizamos o lucro barrando os caloteiros, sem negar crédito excessivo aos bons pagadores.

Realizamos uma análise de sensibilidade variando o limiar de decisão de 0 a 100% para encontrar o ponto de lucro máximo ("Sweet Spot").

![Gráfico de Lucratividade por Threshold](img/analise_lucratividade.png)


**Insights do Gráfico:**
1.  **A Virada do Jogo:** Ao ajustarmos o limiar da Regressão Logística de 0.50 para ~0.22, transformamos um prejuízo de R$ 46.000 em um **Lucro de R$ 121.000**.
2. **Rigor Necessário:** O gráfico mostra que, para este negócio, precisamos ser conservadores. Devemos negar crédito para qualquer cliente com probabilidade de risco acima de 20% a 25%.   
3.  **Robustez:** Robustez: A Regressão Logística (linha laranja) provou ser o modelo mais estável financeiramente, mantendo-se na zona de lucro por uma faixa maior de limiares do que o Random Forest.

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
