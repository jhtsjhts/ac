# Plano do projeto: previsões de hóquei no gelo

## Objetivo

Usar os dados históricos fornecidos em `hockey_development_data/` para prever, para uma época-alvo:

1. A classificação final da época regular das equipas.
2. As equipas que vão mudar de treinador.
3. A terceira tarefa, quando for anunciada.

Cada previsão deve simular o conhecimento disponível **na véspera do primeiro jogo da época-alvo**. Não se podem usar resultados, estatísticas ou decisões que só aconteçam durante ou depois dessa época.

## 1. Confirmar o problema antes de modelar

Registem estas decisões num documento partilhado:

- Qual é a liga e qual é a época-alvo indicadas pelo enunciado completo?
- “Ranking” significa classificação dentro da divisão, conferência ou liga?
- Como se desempata a classificação e que definição de ranking existe nos dados?
- “Mudar de treinador” significa que o treinador principal no fim de uma época não regressa na seguinte, ou inclui qualquer mudança durante a época?
- Que equipas pertencem à população a prever, incluindo equipas novas, relocalizadas ou com mudança de identificador?

O script existente `predict_ranking.py` assume a NHL e 2010, mas confirmem esses valores contra o enunciado. Não os tratem como definitivos sem essa confirmação.

## 2. Inventariar os dados

Os CSV estão em `hockey_development_data/hockey_development_data/`. Comecem por listar, para cada ficheiro:

- colunas, tipos e valores em falta;
- intervalo de anos e número de linhas;
- chaves candidatas, como época, equipa e treinador;
- duplicados e identificadores inconsistentes;
- momento em que cada variável fica disponível.

Tabelas iniciais prováveis:

- `Teams.csv`: resultados e classificação final por equipa e época;
- `Coaches.csv`: treinadores e resultados agregados por equipa e época;
- `TeamSplits.csv`: resultados em casa/fora;
- `Goalies.csv`, `Scoring.csv`: estatísticas individuais que podem ser agregadas por equipa e época;
- `abbrev.csv`: apoio para interpretar abreviaturas.

Inspecionem primeiro o esquema e as chaves antes de combinar tabelas. Evitem carregar ou juntar os CSV grandes sem necessidade.

## 3. Criar uma tabela equipa-época sem fuga de informação

Construam uma tabela canónica com uma linha por equipa e época. Mantenham as colunas de identificação — época, liga, equipa, divisão e conferência — separadas das features e dos alvos.

Para prever a época `t`, usem apenas dados conhecidos até ao fim de `t-1`. Exemplos de features iniciais:

- pontos por jogo, vitórias, golos marcados/sofridos e diferença de golos na época anterior;
- classificação anterior, se comparável à estrutura dessa época;
- tendências calculadas apenas com épocas anteriores;
- desempenho anterior de guarda-redes e treinadores, agregado com cuidado;
- divisão/conferência esperadas para a época-alvo, se forem conhecidas antes do início.

Não usem a classificação final, estatísticas da época ou resultados de treinador da própria época-alvo como features. Uma equipa nova ou uma mudança estrutural exige uma regra explícita para valores sem histórico.

## 4. Definir alvos e baselines

### (a) Classificação

- Confirmem a unidade do ranking e reproduzam a classificação histórica a partir das colunas originais.
- Criem um baseline simples: usar a classificação anterior quando a estrutura permitir; caso contrário, usar uma previsão baseada em pontos/desempenho anteriores.
- Considerem métricas que respeitem a ordem, como erro absoluto médio do lugar e correlação de Spearman. A exatidão exata pode ser reportada como métrica secundária.
- Se o ranking for relativo a uma divisão ou conferência, garantam que as previsões finais são válidas dentro de cada grupo, sem lugares duplicados.

### (b) Mudança de treinador

- Construam o alvo comparando o treinador relevante entre épocas consecutivas, depois de esclarecer a definição com o grupo.
- Não confundam substituições durante uma época com a identidade do treinador principal na data de corte.
- Reportem a taxa-base de mudanças. Como provavelmente haverá menos mudanças do que permanências, usem precision, recall, F1 e PR-AUC, além da matriz de confusão.
- Comparem com um baseline que prevê a classe mais frequente.

## 5. Avaliação temporal

Não façam uma divisão aleatória de linhas: isso pode misturar épocas futuras no treino e produzir uma avaliação pouco realista.

Usem validação progressiva por época. Por exemplo, para avaliar `t`, treinem apenas em épocas anteriores a `t` e validem em `t`; repitam para várias épocas históricas. A seleção de features, imputação, escalamento e escolha de hiperparâmetros devem ser ajustados apenas com os dados de treino de cada divisão temporal.

Escolham o modelo pela média das métricas nas épocas de validação, comparando sempre com o baseline. Guardem uma época final como teste histórico, se houver épocas suficientes, e não a usem para escolher o modelo.

## 6. Modelação e previsões finais

Comecem com modelos simples e interpretableis, como regressão ou árvores pequenas, e só depois comparem com métodos mais complexos. Para cada experimento, guardem:

- épocas de treino e validação;
- features utilizadas e respetivo momento de disponibilidade;
- tratamento de valores em falta;
- modelo e parâmetros;
- métricas por época e média;
- limitações conhecidas.

Depois de escolher o método, treinem com todas as épocas históricas elegíveis anteriores à época-alvo e gerem um ficheiro final com identificador da equipa, época prevista e previsão. Para ranking, incluam também a divisão/conferência e a posição prevista.

## 7. Divisão de trabalho entre três pessoas

### Pessoa 1 — Dados e protocolo de avaliação

- Inventariar os CSV, documentar colunas, chaves e qualidade.
- Definir a tabela equipa-época e harmonizar identificadores.
- Implementar as divisões temporais e os baselines comuns.
- Entregar um dataset reproduzível e um pequeno dicionário de dados.

### Pessoa 2 — Ranking

- Confirmar a definição do ranking e criar o alvo.
- Desenvolver features conhecidas antes da época-alvo.
- Comparar baselines e modelos com métricas ordinais e validação temporal.
- Produzir previsões e explicar erros por época/divisão.

### Pessoa 3 — Mudanças de treinador

- Formalizar a definição do alvo com base no enunciado e nos dados disponíveis.
- Criar features pré-época sem usar informação futura.
- Medir o desequilíbrio das classes e comparar modelos com a taxa-base.
- Produzir previsões, métricas e análise de falsos positivos/negativos.

### Integração conjunta

Todos acordam a chave equipa-época, as regras temporais e as convenções de ficheiros. A Pessoa 1 prepara dados e avaliação comuns; as Pessoas 2 e 3 mantêm pipelines separados para os seus alvos. Quando a tarefa (c) for anunciada, distribuam-na depois de conhecer o alvo e as tabelas relevantes.

## 8. Estrutura sugerida

```text
hockey_development_data/
├── hockey_development_data/   # CSV originais; manter intactos
├── notebooks/                 # exploração e gráficos
├── src/
│   ├── data.py                # leitura, limpeza e tabela equipa-época
│   ├── validation.py          # divisões temporais e métricas
│   ├── ranking.py             # alvo e modelos de ranking
│   └── coaches.py             # alvo e modelos de treinadores
├── outputs/                   # métricas e previsões geradas
└── PLANO_PROJETO.md
```

Não é necessário reorganizar já os ficheiros existentes; esta estrutura é uma proposta para o código novo. Evitem guardar cópias dos CSV originais ou resultados manuais sem registar como foram produzidos.

## 9. Próximos passos concretos

1. Ler o enunciado completo e confirmar época-alvo, liga e definições dos dois alvos.
2. Fazer um inventário de `Teams.csv` e `Coaches.csv`, incluindo anos, colunas e chaves.
3. Produzir uma tabela equipa-época mínima e verificar manualmente algumas épocas.
4. Implementar baselines e validação progressiva por época.
5. Desenvolver os dois modelos em paralelo e comparar resultados com os baselines.
6. Rever cada feature com a pergunta: “Isto era conhecido na véspera do primeiro jogo?”
7. Integrar previsões, métricas, limitações e a terceira tarefa quando for anunciada.

## Observação sobre o código atual

`predict_ranking.py` pode servir como material de exploração, mas a abordagem atual usa variáveis da época que se pretende prever e validação aleatória. Para cumprir a regra temporal do projeto, reconstruam as features a partir de épocas anteriores e avaliem com divisões cronológicas antes de confiar nas previsões geradas.
