# Comparação de modelos para o objetivo b)

Executar a experiência, incluindo a avaliação posterior em 2009:

```powershell
python coach_experiment.py --year 2009 --evaluate-test
```

Para executar apenas a comparação de validação, omitir `--evaluate-test`.
São usadas as bibliotecas já necessárias para `coach_model.py`; não é preciso
instalar novos modelos ou descarregar dados.

São comparadas **12 configurações**, mantendo as mesmas 19 variáveis:

| Dimensão | Valores |
|---|---|
| Algoritmo | Regressão logística, Random Forest, HistGradientBoosting |
| Exemplos usados em cada treino | Todo o histórico, últimas 20 épocas disponíveis |
| Pesos de classe | Sem pesos, `balanced` |

Os parâmetros dos algoritmos ficam fixos e são registados em `protocol.json`.
Na regressão logística mantém-se `C=0.01`, escolhido na validação do modelo
inicial. Esta experiência não procura simultaneamente novos valores de `C` nem
novos parâmetros das árvores. Não existe garantia de melhoria.

A validação usa as mesmas épocas para todas as configurações: **2003, 2005,
2006, 2007 e 2008**, por omissão. Cada configuração é novamente treinada antes
de prever cada época. Não há mistura de equipas de uma época entre treino e
validação.

Para cada uma dessas épocas, há uma divisão anterior para escolher o limiar.
Por exemplo, antes de validar em 2003:

1. Ajustar um modelo com épocas anteriores a 2000.
2. Prever 2000, 2001 e 2002 e escolher dois limiares: um pelo F1 e outro pelo
   F0,5, que dá mais peso à precisão.
3. Ajustar um novo modelo com todas as épocas elegíveis anteriores a 2003 e
   prever 2003 com esses limiares já fixados.

Em ambos os ajustes aplica-se a janela de treino da configuração. As três
épocas para escolher o limiar avançam com a época de validação. A opção
`--calibration-seasons` altera o tamanho desse bloco; o nome refere-se à escolha
do limiar, não à calibração das probabilidades.

O vencedor para cada objetivo é escolhido pela métrica F-beta agregada das
previsões de validação. Os desempates usam average precision média por época,
precision agregada e identificador da configuração. Estes resultados servem
para selecionar configurações e não são um teste independente do vencedor.

Com `--evaluate-test`, a seleção é guardada **antes** de avaliar 2009. Apenas os
dois vencedores escolhidos na validação são avaliados. O limiar de teste é
escolhido com as três últimas épocas anteriores ao teste e há novo treino antes
da previsão. O teste nunca muda o vencedor. Como 2009 já foi consultada durante
o desenvolvimento, a avaliação nessa época é exploratória.

Os resultados ficam em `outputs/coach_experiment/NHL/2009/`:

| Ficheiro | Conteúdo |
|---|---|
| `report.md` | Tabelas e explicação da experiência |
| `comparison.csv` | Métricas agregadas de cada configuração e objetivo |
| `folds.csv` | Métricas e limites de treino por época |
| `validation_predictions.csv` | Todas as previsões de validação |
| `baselines.csv` | Referências nas mesmas épocas de validação |
| `selection.json` | Configurações escolhidas sem usar o teste |
| `protocol.json` | Parâmetros, versões, variáveis e todas as divisões temporais |
| `test_metrics.json` | Avaliação posterior dos vencedores, se pedida |
| `test_predictions_*.csv` | Previsões dos vencedores na época reservada |
| `selected_model_*.joblib` | Pipeline e limiar de cada vencedor ajustados antes do teste |

Na execução efetuada com estes dados, os melhores F1 de validação para cada
algoritmo e janela foram os seguintes (escolhendo o melhor peso de classe em
cada célula e usando limiares escolhidos por F1):

| Algoritmo | Todo o histórico | Últimas 20 épocas |
|---|---|---|
| Regressão logística | 0.286 | 0.317 |
| Random Forest | 0.318 | 0.291 |
| Gradient Boosting | 0.336 | 0.321 |

O vencedor por F1 foi Gradient Boosting com todo o histórico e pesos `balanced`.
Na avaliação posterior de 2009, obteve F1 de **0.214**, precision de **0.120** e
22 falsos positivos. O modelo inicial tinha F1 de 0.231 e 20 falsos positivos:
**esta experiência não melhorou as decisões nesse teste**. A ordenação do risco
nessa época melhorou: ROC AUC de 0.840 e average precision de 0.362, face a 0.519
e 0.128 do modelo inicial. Existem apenas três mudanças observadas nessa época,
pelo que estas diferenças não demonstram uma melhoria estável.

O vencedor por F0,5 foi regressão logística com as últimas 20 épocas e sem pesos.
Na validação, obteve precision de 0.381 e recall de 0.286. Em 2009, previu apenas
uma mudança, que foi falsa, e falhou as três reais. Favorecer a precisão na
seleção reduziu os alertas, mas não trouxe acertos nessa época.

O treino com 20 épocas restringe os exemplos usados no ajuste; variáveis como a
experiência de carreira continuam a usar o passado disponível. A média de F1 e
o desvio-padrão entre épocas estão no CSV. As probabilidades dos modelos com
pesos de classe não foram calibradas.

Uma nova execução substitui os ficheiros produzidos no mesmo diretório. Usar
`--output-dir` permite guardar execuções separadas. Se omitires `--evaluate-test`,
eventuais ficheiros de teste de execuções anteriores não são atualizados; o
relatório e `protocol.json` indicam se houve avaliação de teste na execução.

Verificar os testes:

```powershell
python -m unittest discover -s tests -v
```
