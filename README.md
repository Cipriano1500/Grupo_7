# Deteção de textos humanos e gerados por IA

Trabalho prático de Aprendizagem Profunda: desenvolver e avaliar modelos de classificação binária que distinguem textos humanos (`Human`) de textos gerados por IA (`AI`), de acordo com o [enunciado](./enunciadoTrabalho.pdf).

## Grupo

| Nome | Email |
|---|---|
| Analtina Lopes Tyikongo | analtinalopes13@gmail.com |
| Cipriano Camunga Chitumba | ciprianocchitumba@gmail.com |
| Jemuel Chivinda | — |
| José Chicote Victor | josechicotevictor@gmail.com |

Docentes indicados no enunciado: Miguel Rocha e Lázaro Makili.

## Guia de avaliação

1. Consulte `enunciadoTrabalho.pdf` e `instrucoes.pdf` para os objetivos, tarefas e formato da submissão.
2. Abra os notebooks no Jupyter ou VS Code, seguindo a ordem:
   - `princpl.ipynb` e `tarefa_1.ipynb` — processamento, exploração e classificação da Tarefa 1.
   - `tarefa_2_.ipynb` — Regressão Logística e DNN implementadas com NumPy.
   - `tarefa_3_.ipynb` — classificação com Embedding e BiLSTM em PyTorch.
   - `tarefa_2.ipynb` e `Tarefa_3.ipynb` — notebooks adicionais relacionados com as tarefas.
3. Para avaliar as previsões, abra [`Submissao1/`](./Submissao1/). Os dois ficheiros `submissao1-grupo7-s1.csv` e `submissao1-grupo7-s2.csv` contêm as 100 linhas de `dataset3_inputs.csv`, com as colunas `ID`, `Text` e `output`. A coluna `output` usa exatamente os rótulos `Human` e `AI`, conforme `instrucoes.pdf`.
4. Consulte os pesos e métricas dos modelos em [`modelos_numpy/`](./modelos_numpy/) e [`models/`](./models/). Os scripts de predição e pré-processamento estão na raiz (`Predicao.py`, `predicao1.py` e `pre_proces.py`).
5. Os conjuntos de dados menores estão na raiz: `dataset1_inputs.csv`, `dataset1_outputs.csv`, `dataset2_complete.csv` e `dataset3_inputs.csv`.

Para executar todos os notebooks, execute as células pela ordem apresentada. Alguns dependem de `AI_Human.csv` no diretório raiz. O ficheiro tem mais de 1 GB e não está incluído no GitHub; pode ser obtido [neste link do Google Drive](https://drive.google.com/file/d/16LvnJSI7on6e4wMec5m_2edSnc4zko8c/view), também documentado no início de `tarefa_1.ipynb`. Os grandes ficheiros intermédios de treino e validação gerados a partir dele também não são versionados. Os CSVs de submissão e os resultados dos modelos já gerados estão incluídos para consulta.

## Organização do repositório

| Ficheiro/pasta | Conteúdo |
|---|---|
| `enunciadoTrabalho.pdf`, `instrucoes.pdf` | Enunciado e instruções |
| `princpl.ipynb`, `tarefa_*.ipynb`, `Tarefa_3.ipynb` | Análise, treino e avaliação dos modelos |
| `Submissao1/` | CSVs de submissão do grupo |
| `modelos_numpy/`, `models/`, `clf.pkl`, `tfidf.pkl` | Pesos, modelos, configurações e resultados |
| `dataset*.csv` | Dados de entrada das tarefas |
| `Predicao.py`, `predicao1.py`, `pre_proces.py` | Código de predição e pré-processamento |
| `index.html` | Página de entrada do projeto |
