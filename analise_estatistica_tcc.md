# Análise Estatística Final

## 1. Preparação das avaliações humanas

A avaliação humana foi conduzida com **36 participantes**, considerados como a amostra final do estudo. Cada participante avaliou as legendas produzidas pelos três modelos — GPT-4.1, Llama 4 e SkimCap — nas quatro dimensões do framework ACCR: Accuracy, Completeness, Conciseness e Relevance.

As respostas foram coletadas em escala Likert de 1 a 5 e posteriormente normalizadas para o intervalo de 0 a 100 por meio da transformação Percent of Maximum Possible (POMP):

\[
POMP = \frac{x - 1}{4} \times 100
\]

Assim, os valores da escala foram convertidos da seguinte forma:

| Likert | POMP |
|---:|---:|
| 1 | 0 |
| 2 | 25 |
| 3 | 50 |
| 4 | 75 |
| 5 | 100 |

Para cada combinação entre vídeo, modelo e dimensão, foi calculada a média das avaliações dos 36 participantes. Dessa forma, a análise estatística principal foi conduzida em nível de parágrafo completo, com:

\[
n = 10\ vídeos \times 3\ modelos = 30
\]

observações por dimensão.

---

## 2. Confiabilidade interavaliadores

Antes da comparação entre as avaliações humanas e automáticas, foi verificada a confiabilidade das avaliações humanas por meio do Coeficiente de Correlação Intraclasse, utilizando ICC(2,k), correspondente a um modelo de avaliadores aleatórios com medidas médias.

| Dimensão | ICC(2,k) |
|---|---:|
| Accuracy | 0,9842 |
| Completeness | 0,9803 |
| Conciseness | 0,9701 |
| Relevance | 0,9811 |

Os coeficientes foram elevados em todas as dimensões, indicando forte consistência entre os participantes. Dessa forma, as médias humanas utilizadas como referência apresentam alta confiabilidade interavaliadores.

---

## 3. Concordância entre avaliação humana e ACCR automático

A concordância entre a avaliação humana e o ACCR automático foi mensurada pelos coeficientes de Kendall τb e τc, considerando 30 observações por dimensão.

| Dimensão | n | Kendall τb | Kendall τc | p-valor |
|---|---:|---:|---:|---:|
| Accuracy | 30 | 0,5383 | 0,5381 | < 0,001 |
| Completeness | 30 | 0,6138 | 0,6174 | < 0,001 |
| Conciseness | 30 | 0,5467 | 0,5480 | < 0,001 |
| Relevance | 30 | 0,5450 | 0,5429 | < 0,001 |

Todas as quatro dimensões apresentaram associação positiva e estatisticamente significativa entre as avaliações humanas e os escores produzidos pelo ACCR automático.

A maior concordância foi observada em Completeness, com τb = 0,6138 e τc = 0,6174. As demais dimensões também apresentaram coeficientes superiores a 0,53.

Esses resultados indicam que, de modo geral, legendas que receberam maiores avaliações dos participantes também tenderam a receber maiores escores do avaliador automático. Assim, o ACCR apresenta alinhamento consistente com a percepção humana no conjunto analisado.

Os resultados não devem ser interpretados como evidência de substituição do julgamento humano pelo avaliador automático, mas como evidência de concordância relevante entre os dois métodos.

---

## 4. Concordância entre avaliação humana e métricas tradicionais

A mesma análise de Kendall foi aplicada entre as avaliações humanas e as métricas tradicionais BLEU-4, ROUGE-L, CIDEr, METEOR e R@4.

### 4.1 Accuracy

| Métrica | τb | τc | p-valor |
|---|---:|---:|---:|
| BLEU-4 | -0,2298 | -0,2214 | 0,0895 |
| ROUGE-L | -0,2194 | -0,2192 | 0,0899 |
| CIDEr | -0,2379 | -0,2377 | 0,0659 |
| METEOR | -0,1434 | -0,1431 | 0,2682 |
| R@4 | -0,2248 | -0,1728 | 0,1214 |

Nenhuma métrica apresentou associação estatisticamente significativa com a avaliação humana de Accuracy.

Além disso, BLEU-4, ROUGE-L, CIDEr e METEOR apresentaram coeficientes negativos, embora sejam métricas em que valores maiores representam melhor desempenho.

Esse resultado contrasta com o ACCR automático, que apresentou associação positiva e significativa para Accuracy.

### 4.2 Completeness

| Métrica | τb | τc | p-valor |
|---|---:|---:|---:|
| BLEU-4 | -0,2342 | -0,2262 | 0,0829 |
| ROUGE-L | -0,2189 | -0,2189 | 0,0900 |
| CIDEr | -0,2834 | -0,2835 | 0,0282 |
| METEOR | -0,1246 | -0,1244 | 0,3351 |
| R@4 | -0,2915 | -0,2247 | 0,0441 |

CIDEr apresentou associação negativa com a avaliação humana de Completeness. Como CIDEr é uma métrica em que valores maiores representam melhor desempenho, a direção negativa indica divergência em relação à avaliação humana.

O R@4 também apresentou associação negativa, porém sua interpretação é distinta, pois valores menores indicam menor repetição e, portanto, melhor desempenho. Nesse caso, uma correlação negativa pode representar concordância de direção.

### 4.3 Conciseness

| Métrica | τb | τc | p-valor |
|---|---:|---:|---:|
| BLEU-4 | -0,1806 | -0,1738 | 0,1825 |
| ROUGE-L | -0,2266 | -0,2269 | 0,0802 |
| CIDEr | -0,1850 | -0,1852 | 0,1532 |
| METEOR | -0,1227 | -0,1227 | 0,3439 |
| R@4 | -0,3472 | -0,2667 | 0,0169 |

O resultado mais relevante para Conciseness foi observado com R@4. Como essa métrica mede repetição e valores menores representam menor redundância, a correlação negativa é coerente com a interpretação de que maior repetição tende a estar relacionada a menor concisão.

Esse comportamento sugere que R@4 pode capturar um aspecto específico de redundância textual, embora não funcione como medida geral de qualidade semântica.

### 4.4 Relevance

| Métrica | τb | τc | p-valor |
|---|---:|---:|---:|
| BLEU-4 | -0,2471 | -0,2381 | 0,0679 |
| ROUGE-L | -0,2564 | -0,2565 | 0,0475 |
| CIDEr | -0,1963 | -0,1964 | 0,1292 |
| METEOR | -0,1480 | -0,1479 | 0,2531 |
| R@4 | -0,2762 | -0,2123 | 0,0571 |

ROUGE-L apresentou associação negativa com a avaliação humana de Relevance. Como valores maiores de ROUGE-L representam maior sobreposição lexical com as referências, esse resultado indica que maior similaridade lexical não necessariamente esteve associada a maior relevância percebida pelos participantes.

---

## 5. Comparação do alinhamento com o julgamento humano

A comparação entre os coeficientes obtidos pelo ACCR e pelas métricas tradicionais evidencia diferenças claras entre os dois paradigmas de avaliação.

| Dimensão | Humano × ACCR τb | Comportamento das métricas tradicionais |
|---|---:|---|
| Accuracy | +0,5383 | Fracas e não significativas |
| Completeness | +0,6138 | Predominantemente fracas ou negativas |
| Conciseness | +0,5467 | Predominantemente fracas |
| Relevance | +0,5450 | Predominantemente fracas ou negativas |

Enquanto o ACCR apresentou associações positivas e estatisticamente significativas em todas as dimensões, as métricas tradicionais não acompanharam o julgamento humano com a mesma consistência.

Esse resultado fornece evidências favoráveis à hipótese de que o ACCR apresenta maior concordância com a percepção humana do que as métricas tradicionais avaliadas neste estudo.

---

## 6. Relação direta entre ACCR automático e métricas tradicionais

A correlação direta entre o ACCR automático e as métricas tradicionais foi utilizada para avaliar o grau de convergência ou divergência entre os dois paradigmas de avaliação.

### 6.1 Accuracy

| Métrica | τb | τc | p-valor |
|---|---:|---:|---:|
| BLEU-4 | -0,0152 | -0,0143 | 0,9123 |
| ROUGE-L | -0,0119 | -0,0119 | 0,9285 |
| CIDEr | -0,0735 | -0,0738 | 0,5780 |
| METEOR | 0,0570 | 0,0571 | 0,6666 |
| R@4 | -0,3659 | -0,2741 | 0,0136 |

### 6.2 Completeness

| Métrica | τb | τc | p-valor |
|---|---:|---:|---:|
| BLEU-4 | 0,0025 | 0,0024 | 0,9854 |
| ROUGE-L | -0,0190 | -0,0191 | 0,8858 |
| CIDEr | -0,0855 | -0,0862 | 0,5182 |
| METEOR | 0,0784 | 0,0790 | 0,5536 |
| R@4 | -0,3565 | -0,2667 | 0,0164 |

### 6.3 Conciseness

| Métrica | τb | τc | p-valor |
|---|---:|---:|---:|
| BLEU-4 | -0,1270 | -0,1197 | 0,3588 |
| ROUGE-L | -0,1424 | -0,1436 | 0,2815 |
| CIDEr | 0,0522 | 0,0526 | 0,6929 |
| METEOR | -0,0879 | -0,0885 | 0,5066 |
| R@4 | -0,3168 | -0,2370 | 0,0328 |

### 6.4 Relevance

| Métrica | τb | τc | p-valor |
|---|---:|---:|---:|
| BLEU-4 | 0,0280 | 0,0262 | 0,8398 |
| ROUGE-L | 0,0143 | 0,0143 | 0,9141 |
| CIDEr | -0,0095 | -0,0095 | 0,9427 |
| METEOR | 0,0810 | 0,0810 | 0,5411 |
| R@4 | -0,3639 | -0,2716 | 0,0144 |

BLEU-4, ROUGE-L, CIDEr e METEOR apresentaram correlações próximas de zero e predominantemente não significativas em relação ao ACCR.

O R@4 apresentou associações negativas significativas nas quatro dimensões. Como valores menores de R@4 representam menor repetição, esse resultado possui direção compatível com escores maiores de qualidade no ACCR.

De forma geral, os resultados indicam baixa convergência entre o ACCR e as métricas tradicionais baseadas em sobreposição lexical.

---

## 7. Comparação dos três modelos segundo a avaliação humana

O desempenho de GPT-4.1, Llama 4 e SkimCap foi comparado por meio do teste de Friedman, utilizando os 10 vídeos como blocos pareados.

| Variável | n | χ² | p-valor | Kendall's W |
|---|---:|---:|---:|---:|
| Accuracy | 10 | 16,20 | < 0,001 | 0,810 |
| Completeness | 10 | 16,20 | < 0,001 | 0,810 |
| Conciseness | 10 | 17,90 | < 0,001 | 0,895 |
| Relevance | 10 | 16,80 | < 0,001 | 0,840 |
| ACCR médio | 10 | 17,90 | < 0,001 | 0,895 |

Foram observadas diferenças estatisticamente significativas entre os três modelos em todas as dimensões e também no ACCR médio.

Os valores elevados de Kendall's W indicam que as diferenças observadas entre os modelos possuem magnitude relevante no conjunto analisado.

---

## 8. Comparações post-hoc na avaliação humana

Após os resultados significativos do teste de Friedman, foram realizadas comparações pareadas por meio do teste de postos sinalizados de Wilcoxon, com correção de Holm.

### 8.1 Accuracy

| Comparação | p Holm | Resultado |
|---|---:|---|
| GPT-4.1 × Llama 4 | 0,0117 | Significativa |
| GPT-4.1 × SkimCap | 0,0059 | Significativa |
| Llama 4 × SkimCap | 0,0117 | Significativa |

### 8.2 Completeness

| Comparação | p Holm | Resultado |
|---|---:|---|
| GPT-4.1 × Llama 4 | 0,0273 | Significativa |
| GPT-4.1 × SkimCap | 0,0059 | Significativa |
| Llama 4 × SkimCap | 0,0078 | Significativa |

### 8.3 Conciseness

| Comparação | p Holm | Resultado |
|---|---:|---|
| GPT-4.1 × Llama 4 | 0,0506 | Não significativa |
| GPT-4.1 × SkimCap | 0,0059 | Significativa |
| Llama 4 × SkimCap | 0,0059 | Significativa |

A comparação entre GPT-4.1 e Llama 4 em Conciseness ficou próxima do limiar de significância, porém não atingiu α = 0,05 após a correção de Holm.

### 8.4 Relevance

| Comparação | p Holm | Resultado |
|---|---:|---|
| GPT-4.1 × Llama 4 | 0,0371 | Significativa |
| GPT-4.1 × SkimCap | 0,0059 | Significativa |
| Llama 4 × SkimCap | 0,0059 | Significativa |

### 8.5 ACCR médio humano

| Comparação | p Holm | Resultado |
|---|---:|---|
| GPT-4.1 × Llama 4 | 0,0382 | Significativa |
| GPT-4.1 × SkimCap | 0,0059 | Significativa |
| Llama 4 × SkimCap | 0,0059 | Significativa |

Considerando o ACCR médio humano, a ordenação observada foi:

\[
\boxed{\text{GPT-4.1} > \text{Llama 4} > \text{SkimCap}}
\]

Essa ordenação também foi observada de forma geral nas quatro dimensões individuais, com exceção da diferença entre GPT-4.1 e Llama 4 em Conciseness, que não foi estatisticamente significativa após a correção de Holm.

---

## 9. Comparação dos três modelos segundo o ACCR automático

O teste de Friedman também identificou diferenças significativas entre os modelos nas quatro dimensões avaliadas automaticamente.

| Dimensão | n | χ² | p-valor | Kendall's W |
|---|---:|---:|---:|---:|
| Accuracy | 10 | 12,81 | 0,0017 | 0,641 |
| Completeness | 10 | 12,05 | 0,0024 | 0,603 |
| Conciseness | 10 | 13,35 | 0,0013 | 0,668 |
| Relevance | 10 | 12,81 | 0,0017 | 0,641 |

O avaliador automático conseguiu identificar diferenças globais entre os três modelos em todas as dimensões.

### 9.1 Wilcoxon com correção de Holm

#### Accuracy

| Comparação | p Holm | Resultado |
|---|---:|---|
| GPT-4.1 × Llama 4 | 0,0547 | Não significativa |
| GPT-4.1 × SkimCap | 0,0229 | Significativa |
| Llama 4 × SkimCap | 0,0547 | Não significativa |

#### Completeness

| Comparação | p Holm | Resultado |
|---|---:|---|
| GPT-4.1 × Llama 4 | 0,0921 | Não significativa |
| GPT-4.1 × SkimCap | 0,0229 | Significativa |
| Llama 4 × SkimCap | 0,0229 | Significativa |

#### Conciseness

| Comparação | p Holm | Resultado |
|---|---:|---|
| GPT-4.1 × Llama 4 | 0,1410 | Não significativa |
| GPT-4.1 × SkimCap | 0,0059 | Significativa |
| Llama 4 × SkimCap | 0,0478 | Significativa |

#### Relevance

| Comparação | p Holm | Resultado |
|---|---:|---|
| GPT-4.1 × Llama 4 | 0,0797 | Não significativa |
| GPT-4.1 × SkimCap | 0,0231 | Significativa |
| Llama 4 × SkimCap | 0,0547 | Não significativa |

O ACCR automático diferenciou de forma consistente GPT-4.1 de SkimCap, mas apresentou menor capacidade de distinguir GPT-4.1 de Llama 4.

Esse comportamento é semelhante à tendência geral observada pelos humanos, embora os participantes tenham discriminado os dois modelos generativos com maior frequência.

---

## 10. Comparação dos modelos segundo métricas tradicionais

Os resultados do teste de Friedman para as métricas tradicionais foram:

| Métrica | n | χ² | p-valor | Kendall's W | Resultado |
|---|---:|---:|---:|---:|---|
| BLEU-4 | 10 | 4,65 | 0,0979 | 0,232 | Não significativa |
| ROUGE-L | 10 | 9,80 | 0,0074 | 0,490 | Significativa |
| CIDEr | 10 | 3,80 | 0,1496 | 0,190 | Não significativa |
| METEOR | 10 | 7,40 | 0,0247 | 0,370 | Significativa |
| R@4 | 10 | 3,13 | 0,2090 | 0,157 | Não significativa |

BLEU-4, CIDEr e R@4 não detectaram diferenças globais significativas entre os três modelos.

ROUGE-L e METEOR apresentaram diferenças globais significativas e, portanto, foram submetidos ao pós-teste de Wilcoxon com correção de Holm.

### 10.1 ROUGE-L

| Comparação | Mediana A | Mediana B | p Holm | Resultado |
|---|---:|---:|---:|---|
| GPT-4.1 × Llama 4 | 25,10 | 21,92 | 0,6250 | Não significativa |
| GPT-4.1 × SkimCap | 25,10 | 33,16 | 0,0117 | Significativa |
| Llama 4 × SkimCap | 21,92 | 33,16 | 0,0273 | Significativa |

ROUGE-L favoreceu significativamente o SkimCap em relação a GPT-4.1 e Llama 4.

Esse resultado apresenta uma ordenação oposta à observada na avaliação humana, na qual SkimCap recebeu os menores escores.

### 10.2 METEOR

| Comparação | Mediana A | Mediana B | p Holm | Resultado |
|---|---:|---:|---:|---|
| GPT-4.1 × Llama 4 | 21,20 | 27,21 | 0,9219 | Não significativa |
| GPT-4.1 × SkimCap | 21,20 | 35,58 | 0,1465 | Não significativa |
| Llama 4 × SkimCap | 27,21 | 35,58 | 0,1465 | Não significativa |

Embora METEOR tenha apresentado diferença global significativa no teste de Friedman, nenhuma comparação pareada permaneceu significativa após a correção de Holm.

---

## 11. Relação dos resultados com os objetivos do estudo

### Objetivo i — comparar os paradigmas de avaliação

A correlação direta entre ACCR e as métricas tradicionais mostrou baixa convergência para BLEU-4, ROUGE-L, CIDEr e METEOR.

Esse resultado indica que os dois paradigmas avaliam propriedades diferentes das legendas. Enquanto as métricas tradicionais dependem fortemente da correspondência lexical com as referências, o ACCR realiza uma avaliação multidimensional orientada a aspectos semânticos.

O comportamento do R@4 foi parcialmente distinto, apresentando associações negativas compatíveis com sua natureza de métrica de repetição.

### Objetivo ii — verificar se modelos generativos superam o modelo dedicado

A avaliação humana encontrou diferenças significativas entre os três modelos e sustentou, no ACCR médio, a seguinte ordenação:

\[
\text{GPT-4.1} > \text{Llama 4} > \text{SkimCap}
\]

O ACCR automático também identificou diferenças globais entre os modelos nas quatro dimensões e diferenciou de forma consistente GPT-4.1 de SkimCap.

As métricas tradicionais, entretanto, não reproduziram essa ordenação de maneira consistente. BLEU-4, CIDEr e R@4 não encontraram diferenças globais significativas, enquanto ROUGE-L favoreceu o SkimCap em relação aos dois modelos generativos.

### Objetivo iii — avaliar a validade dos métodos automáticos frente ao julgamento humano

O ACCR apresentou correlações positivas e significativas com os humanos em todas as dimensões:

- Accuracy: τb = 0,5383;
- Completeness: τb = 0,6138;
- Conciseness: τb = 0,5467;
- Relevance: τb = 0,5450.

As métricas tradicionais apresentaram associações substancialmente menores, predominantemente fracas, não significativas ou em direção contrária à qualidade percebida.

Assim, no conjunto analisado, o ACCR apresentou maior alinhamento com o julgamento humano do que as métricas tradicionais.

---

## 12. Síntese dos resultados

Os resultados estatísticos apontam três tendências principais.

Primeiro, as avaliações humanas apresentaram elevada confiabilidade interavaliadores, com ICC(2,k) superior a 0,97 em todas as dimensões. Isso sustenta o uso das médias humanas como referência para a validação dos métodos automáticos.

Segundo, o ACCR automático apresentou concordância consistente com o julgamento humano. Os coeficientes de Kendall foram positivos e estatisticamente significativos nas quatro dimensões, variando aproximadamente entre 0,54 e 0,61.

Terceiro, as métricas tradicionais não reproduziram o julgamento humano com a mesma consistência. A maioria das correlações foi fraca e não significativa, enquanto algumas métricas apresentaram direção contrária à percepção dos participantes.

A divergência também foi observada na comparação entre modelos. Os participantes humanos favoreceram GPT-4.1, seguido por Llama 4 e SkimCap. Entretanto, ROUGE-L atribuiu valores significativamente maiores ao SkimCap em relação aos dois modelos generativos.

Esse contraste indica que maior sobreposição lexical com as referências não implica necessariamente maior qualidade semântica percebida.

---

## 13. Considerações sobre a interpretação estatística

Embora tenham sido utilizadas 30 observações por dimensão nas análises de correlação, a comparação pareada entre modelos utiliza apenas 10 vídeos como blocos no teste de Friedman e nos testes de Wilcoxon.

Por esse motivo, os resultados de comparação entre modelos devem ser interpretados de forma exploratória, conforme previsto na metodologia do estudo.

A interpretação não deve se limitar ao valor-p. Também devem ser considerados:

- a magnitude das diferenças observadas;
- o valor de Kendall's W no teste de Friedman;
- as medianas obtidas pelos modelos;
- a consistência do padrão entre as diferentes dimensões e métodos de avaliação.

Além disso, é necessário considerar a direção específica do R@4. Diferentemente de BLEU-4, ROUGE-L, CIDEr e METEOR, valores menores de R@4 representam melhor desempenho por indicarem menor repetição de 4-gramas. Assim, uma correlação negativa entre R@4 e os escores humanos ou ACCR pode representar concordância, e não divergência.

---

## 14. Conclusão da análise estatística

Os resultados fornecem evidências de que a avaliação semântica automática baseada no ACCR apresenta maior alinhamento com o julgamento humano do que as métricas automáticas tradicionais consideradas neste estudo.

As quatro dimensões do ACCR apresentaram correlações positivas e estatisticamente significativas com as avaliações humanas, enquanto BLEU-4, ROUGE-L, CIDEr e METEOR apresentaram associações predominantemente fracas, não significativas ou, em alguns casos, em direção contrária à qualidade percebida.

A avaliação humana também identificou diferenças consistentes entre os modelos, com GPT-4.1 apresentando o melhor desempenho geral, seguido por Llama 4 e SkimCap.

O ACCR automático reproduziu parcialmente essa tendência, diferenciando claramente GPT-4.1 de SkimCap, embora tenha apresentado menor capacidade de distinção entre os dois modelos generativos.

As métricas tradicionais, por sua vez, não reproduziram de forma consistente a ordenação humana. O caso mais evidente foi o ROUGE-L, que favoreceu significativamente o SkimCap em relação a GPT-4.1 e Llama 4.

Dessa forma, os resultados reforçam a importância de métodos de avaliação que considerem propriedades semânticas e multidimensionais das legendas, especialmente em tarefas de video paragraph captioning, nas quais descrições semanticamente adequadas podem apresentar baixa sobreposição lexical com as referências.
