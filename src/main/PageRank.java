package main;

import java.util.*;
import java.io.*;

import mpi.*;

public class PageRank {

	public static void main(String args[]) throws Exception {

		/* Declaração de variáveis e estruturas do PageRank */
		HashMap<Integer, ArrayList<Integer>> dados = new HashMap<>();
		ArrayList<Integer> ligacoesNo = new ArrayList<>();
		String filename = args[3];
		int interacoes = Integer.parseInt(args[4]);
		double limite = Double.parseDouble(args[5]);
		long inicioEntrada, finalEntrada = 0, inicioExecucao = 0;

		/* Inicialização do MPI */
		MPI.Init(args);

		int rank = MPI.COMM_WORLD.Rank();

		/* Leitura dos dados através do arquivo para cada processo*/
		inicioEntrada = System.currentTimeMillis();

		leituraDados(filename, dados, ligacoesNo);

		if(rank == 0) {

			finalEntrada = System.currentTimeMillis() - inicioEntrada;

		}

		int numeroNos = ligacoesNo.size() / 2;
		int totalNumeroNos = calcularNumeroNosTotais(numeroNos);
		double pageRanks[] = new double[totalNumeroNos];

		if(rank == 0) {

			inicioExecucao = System.currentTimeMillis();

		}

		//Inicia a computação do pagerank para os nós
		pageRank(dados, ligacoesNo, totalNumeroNos, interacoes, limite, pageRanks);

		//Salvar o resultado em um arquivo e printa tempos de execução
		salvarResultado(finalEntrada, inicioExecucao, rank, pageRanks);

		MPI.Finalize();

	}

	private static void salvarResultado(long finalEntrada, long inicioExecucao, int rank, double[] pageRanks) throws IOException {

		if(rank != 0) {

			return;

		}

		LinkedHashMap<Integer, Double> pageRanksMap = new LinkedHashMap<>();
		long finalExecucao = System.currentTimeMillis() - inicioExecucao;

		//PageRanks transformado para Map para facilitar escrita no arquivo
		for(int i = 0; i < pageRanks.length; i++) {

			pageRanksMap.put(i, pageRanks[i]);

		}

		long inicioEscrita = System.currentTimeMillis();

		escreverArquivo(pageRanksMap);

		long finalEscrita = System.currentTimeMillis() - inicioEscrita;

		System.out.println("Resultado gravado no arquivo: saida.txt");
		System.out.println("Tempo total para operações de I/O: " + (finalEntrada + finalEscrita) + " milissegundos");
		System.out.println("Tempo total tomado pela execução do algoritmo Page Rank: " + finalExecucao + " milissegundos");

	}

	//Calcula numero de nós totais a partir do numéro de nós de cada bloco
	private static int calcularNumeroNosTotais(int numeroNosPorProcessos) {

		int numeroNosArray[] = new int[]{numeroNosPorProcessos};
		int totalNumeroNosArray[] = new int[1];

		MPI.COMM_WORLD.Allreduce(numeroNosArray, 0, totalNumeroNosArray, 0, 1, MPI.INT, MPI.SUM);

		return totalNumeroNosArray[0];

	}

	private static void leituraDados(String filename, HashMap<Integer, ArrayList<Integer>> dados, ArrayList<Integer> ligacoesNo) throws IOException {

		if(MPI.COMM_WORLD.Rank() == 0) {

			//preenchimento dos nós conforme dispostos no arquivo.
			lerArquivo(filename, dados);

			ArrayList<Integer> nos = new ArrayList<>(dados.keySet());
			int totalNos = dados.size();
			int numeroProcessos = MPI.COMM_WORLD.Size();
			int divisoes = (totalNos) / numeroProcessos;
			int resto = (totalNos) % numeroProcessos;
			int strtIndex = 0, j = 0, inicio, h;
			int tamBloco, tamLigacoes;

			for(int i = 0; i < numeroProcessos; i++) {

				tamBloco = calcularTamanhoBloco(divisoes, resto, i);
				tamLigacoes = tamBloco * 2;

				//vetor de ligações dos nós
				int[] ligacoes = new int[tamLigacoes];

				h = 0;
				inicio = strtIndex;

				//repetição para acessar os valores dos nós a cada bloco
				for(strtIndex = inicio; strtIndex <= ((inicio + tamBloco) - 1); strtIndex++) {

					//atribuição ao vetor de ligações, determinando o nó e posteriormente seu número de ligações
					int no = nos.get(j++);
					ligacoes[h++] = no;
					ligacoes[h++] = dados.get(no).size();

				}

				preencherNosDestinoPorProcesso(dados, tamLigacoes, i, ligacoes, ligacoesNo);

			}

		} else {

			receberNosProcessoMestre(dados, ligacoesNo);

		}

	}

	private static void receberNosProcessoMestre(HashMap<Integer, ArrayList<Integer>> dados, ArrayList<Integer> ligacoesNo) {

		int[] tamLigacoesComm = new int[1];

		MPI.COMM_WORLD.Recv(tamLigacoesComm, 0, 1, MPI.INT, 0, 0);

		int tamLigacoes = tamLigacoesComm[0];
		int[] ligacoes = new int[tamLigacoes];

		MPI.COMM_WORLD.Recv(ligacoes, 0, tamLigacoes, MPI.INT, 0, 1);

		for(int l = 0; l < tamLigacoes; l++) {

			ligacoesNo.add(ligacoes[l]);

		}

		for(int k = 0; k < tamLigacoes; k = k + 2) {

			int no = ligacoes[k];
			int numeroLigacoesNo = ligacoes[k + 1];
			int[] nosDestinoArray = new int[numeroLigacoesNo];
			ArrayList<Integer> nosDestino = new ArrayList<>();

			MPI.COMM_WORLD.Recv(nosDestinoArray, 0, numeroLigacoesNo, MPI.INT, 0, 2);

			for (int m = 0; m < numeroLigacoesNo; m++) {

				nosDestino.add(nosDestinoArray[m]);

			}

			dados.put(no, nosDestino);

		}

	}

	private static void preencherNosDestinoPorProcesso(HashMap<Integer, ArrayList<Integer>> dados, int tamLigacoes, int i, int[] ligacoes, ArrayList<Integer> ligacoesNo) {

		//verificação para o primeiro processo
		if(i == 0) {

			//preenchimento dos valores de ligações a serem usados além do método
			for(int l = 0; l < tamLigacoes; l++) {

				ligacoesNo.add(ligacoes[l]);

			}

		} else {

			int[] tamDados = new int[]{tamLigacoes};

			//envio do tamanho dos dados para os demais processos
			MPI.COMM_WORLD.Send(tamDados, 0, 1, MPI.INT, i, 0);
			MPI.COMM_WORLD.Send(ligacoes, 0, tamLigacoes, MPI.INT, i, 1);

		}

		for(int k = 0; k < tamLigacoes; k = k + 2) {

			if(i != 0) {

				int no = ligacoes[k];
				int numeroLigacoesNo = ligacoes[k + 1];
				int[] nosDestino = new int[numeroLigacoesNo];

				for(int n = 0; n < numeroLigacoesNo; n++) {

					nosDestino[n] = dados.get(no).get(n);

				}

				MPI.COMM_WORLD.Send(nosDestino, 0, numeroLigacoesNo, MPI.INT, i, 2);

			}

		}

	}

	//Calcula o tamanho do bloco, para que cada processo assuma este tamanho
	private static int calcularTamanhoBloco(int divisoes, int resto, int i) {

		return resto == 0 ? divisoes : (i < resto) ? (divisoes + 1) : (divisoes);

	}

	private static void lerArquivo(String filename, HashMap<Integer, ArrayList<Integer>> dados) throws IOException {

		DataInputStream dataInputStream = new DataInputStream(new FileInputStream("files/" + filename));
		BufferedReader leitorUrls = new BufferedReader(new InputStreamReader(dataInputStream));
		String entrada;

		while((entrada = leitorUrls.readLine()) != null) {

			ArrayList<Integer> nosDestino = new ArrayList<>();

			//Declaração de array dos nós
			String[] nos = entrada.split(" ");

			//Preenchimento dos nós destino do primeiro nó da linha
			for(int i = 1; i < nos.length; i++) {

				nosDestino.add(Integer.valueOf(nos[i]));

			}

			//relação dos nós destino com o primeiro nó da linha
			dados.put(Integer.valueOf(nos[0]), nosDestino);

		}

		dataInputStream.close();

	}

	private static void pageRank(HashMap<Integer, ArrayList<Integer>> dados, ArrayList<Integer> ligacoesNo, int totalNumeroNos,
								 int interacoes, double limite, double[] pageRanks) {

		//Probabilidade inicial para o PageRank computado iterativamente
		for(int k = 0; k < totalNumeroNos; k++) {

			pageRanks[k] = 1.0 / totalNumeroNos;

		}

		double fatorCompensacao, fatorCompensacaoTotal, fatorAmortecimento = 0.85;
		int no, numeroLigacoesNo, noDestino, l = 0;
		ArrayList<Integer> nosDestino;

		double[] pageRanksPorInteracao = new double[totalNumeroNos];
		double[] pageRanksProcessoAtual = new double[totalNumeroNos];
		double[] variacao = new double[]{0.0};

		int rank = MPI.COMM_WORLD.Rank();

		if(rank == 0) {

			System.out.println("Máximo de interações: " + interacoes + ", limite da variação: " + limite);

		}

		// Algoritmo interativo
		do {

			fatorCompensacao = 0.0;

			//Calcula o PageRank local
			for(int i = 0; i < ligacoesNo.size(); i = i + 2) {

				no = ligacoesNo.get(i);
				nosDestino = dados.get(no);
				numeroLigacoesNo = nosDestino.size();

				for(int j = 0; j < numeroLigacoesNo; j++) {

					noDestino = nosDestino.get(j);
					pageRanksProcessoAtual[noDestino] = pageRanksProcessoAtual[noDestino] + pageRanks[no] / (double) numeroLigacoesNo;

				}

				//Caso seja um nó sem ligação, é incrementado o valor de amortecimento
				if(numeroLigacoesNo == 0) {

					fatorCompensacao += pageRanks[no];

				}

			}

			//Soma os page ranks de cada bloco ao pagerank total
			MPI.COMM_WORLD.Allreduce(pageRanksProcessoAtual, 0, pageRanks, 0, totalNumeroNos, MPI.DOUBLE, MPI.SUM);

			//Soma os fatores de compensação de cada bloco ao fator de compensação total
			double[] fatorCompensacaoArray = new double[1];
			double[] fatorCompensacaoTotalArray = new double[1];
			fatorCompensacaoArray[0] = fatorCompensacao;
			MPI.COMM_WORLD.Allreduce(fatorCompensacaoArray, 0, fatorCompensacaoTotalArray, 0, 1, MPI.DOUBLE, MPI.SUM);
			fatorCompensacaoTotal = fatorCompensacaoTotalArray[0];

			// Recalcula os page ranks considerando o fator de amortecimento padrão (0.85) e o fator de compensação por nós
			if(rank == 0) {

				double fatorCompensacaoPorNo = fatorCompensacaoTotal / totalNumeroNos;

				for(int i = 0; i < totalNumeroNos; i++) {

					pageRanks[i] = pageRanks[i] + fatorCompensacaoPorNo;

				}

				for(int i = 0; i < totalNumeroNos; i++) {

					pageRanks[i] = fatorAmortecimento * pageRanks[i] + ((1 - fatorAmortecimento) / (double) totalNumeroNos);

				}

				variacao[0] = 0.0;

				//Calculo de variação da interação atual com o valor total do pageranks com a finalidade
				//de determinar se as interações devem ou não continuar
				for(int i = 0; i < totalNumeroNos; i++) {

					variacao[0] += Math.abs(pageRanksPorInteracao[i] - pageRanks[i]);
					pageRanksPorInteracao[i] = pageRanks[i];

				}

			}

			//envia a todos processos os valores de variação e atualiza o pageRanks
			MPI.COMM_WORLD.Bcast(variacao, 0, 1, MPI.DOUBLE, 0);
			MPI.COMM_WORLD.Bcast(pageRanks, 0, totalNumeroNos, MPI.DOUBLE, 0);

			//Reiniciar valores do pageRanks auxiliar
			for(int k = 0; k < totalNumeroNos; k++) {

				pageRanksProcessoAtual[k] = 0.0;

			}

			if(rank == 0) {

				System.out.println("Interação " + l + " variação: " + variacao[0]);

			}

		//A Variação deve ser menor que o limite e as interações menores do que o número informado para o máximo de interações
		} while(variacao[0] > limite && ++l < interacoes);

	}

	private static void escreverArquivo(LinkedHashMap<Integer, Double> pageRanksMap) throws IOException {

		int[] nos = new int[pageRanksMap.size()];
		double[] pageRanks = new double[pageRanksMap.size()];
		int j = 0;

		for(Map.Entry<Integer, Double> mapEntry : pageRanksMap.entrySet()) {

			nos[j] = mapEntry.getKey();
			pageRanks[j] = mapEntry.getValue();
			j++;

		}

		//Ordenação dos pageRanks
		List<Double> pageRanksOrdenados = new ArrayList<>(pageRanksMap.values());
		Collections.sort(pageRanksOrdenados);
		ListIterator iterator = pageRanksOrdenados.listIterator(pageRanksOrdenados.size());
		int numeroNos = 0;
		int numeroTotalNos = pageRanksOrdenados.size();

		Writer output = new BufferedWriter(new FileWriter(new File("saida.txt")));
		output.append("\nTop 10 de páginas de maiores valores de 'page rank' " + "\n\n" + "--------------------------------------" + "\n");
		output.append("|\t" + "URL" + "\t\t|\t" + "Page Rank" + "\t\t\t|\n" + "--------------------------------------" + "\n");

		//Escrita do Top 10 de páginas de maiores valores de 'page rank'
		while(iterator.hasPrevious() && numeroNos++ < 10) {

			double pageRankValor = Double.valueOf(iterator.previous().toString());

			for(int i = 0; i < numeroTotalNos; i++) {

				if(pageRanks[i] == pageRankValor) {

					output.write("|\t" + nos[i] + "\t\t|\t" + String.format("%2.17f", pageRankValor) + "\t|\n");
					System.out.println("|\t" + nos[i] + "\t\t|\t" + String.format("%2.17f", pageRankValor) + "\t|\n");
					output.write("---------------------------------------" + "\n");
					pageRanks[i] = -1;

					break;

				}

			}

		}

		output.close();

	}

}