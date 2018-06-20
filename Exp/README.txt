Seleção automática de features com GP e classificação com KNN

	- Dados na pasta 'data':
		dataset 1: Dados do canal central EEG
		dataset 2: Dados dos 3 canais EEG
		dataset 3: Dados das 26 PCAs
		dataset 4: Dados dos 3 canais para pacientes homens
		dataset 5: Dados dos 3 canais para pacientes mulheres

	Parâmetros via terminal:
	-gen <int>: Número de gerações
	-pop <int>: Número de indivíduos
	-depth <int>: Tamanho máximo de árvore
	-k <int>: Parâmetro K do KNN
	-execs <int> <int>: Range das execuções, por exemplo, 1 a 10
	-dataset <int>: Número do dataset
	-path <string>: Nome da pasta onde serão salvas as informações da execução
	-fileID <string>: String a ser adicionada aos nomes dos arquivos de log gerados
	-v <int>: Nível de verbose. 0 -> não imprime em tela. 1 -> informações entre execuções. 2->informações de toas as gerações de todas as execuções.
	-optmize <int> <int> <string>: Número de variáveis que se deseja otimizar depois peso da variável e nome da variável e assim sucessivamente. Variáveis possíveis: acc, prec_S, prec_NS, rec_S, rec_NS, f1_S, f1_NS, TN, FP, FN, TP.

	Hiper-parametros: (VER NO CÓDIGO, ESSES SÃO OS QUE ACHO MAIS CRUCIAIS)
	-ini: Método de inicialização
	-sel: Método de seleção
	-mut: Método de mutação
	-crs: Método de crossover

	Se executar somente o programa: python3 main.py
	Será realizado o teste default com os seguntes parâmetros
	-gen 100
	-pop 100
	-depth 10
	-k 5
	-execs 1 10
	-dataset 1
	-path Default_Try/
	-fileID Default
	-v 1
	-optmize 1 1 acc

Exemplo: Execução em segundo plano em 1 core com nohup

nohup python3 main.py -v 0 -gen 500 -pop 100 -depth 40 -k 3 -execs 1 10 -dataset 5 -path TESTE/ -fileID Teste1 -optmize 2 1 TP 1 TN &
