package main;

import java.util.*;
import java.io.*;

import mpi.*;

public class PageRank {

	public static void main(String args[]) throws Exception {

		/* Declaração de variáveis e estruturas do PageRank */
		HashMap<Integer, ArrayList<Integer>> dados = new HashMap<>();
		ArrayList<Integer> indiceDados = new ArrayList<>();
		LinkedHashMap<Integer, Double> pageRanks = new LinkedHashMap<>();
		String filename = args[3], outfilename = "saida.txt";
		double limite = Double.parseDouble(args[5]);
		int totalUrls, numeroUrls, interacoes = Integer.parseInt(args[4]);
		long inicioEntrada, finalEntrada = 0, inicioExecucao = 0, finalExecucao = 0, inicioSaida, finalSaida;

		/* Inicialização do MPI */
		MPI.Init(args);

		int rank = MPI.COMM_WORLD.Rank();

		/* Leitura dos dados através do arquivo para cada processo*/
		inicioEntrada = System.currentTimeMillis();

		leituraDados(filename, dados, indiceDados);

		if(rank == 0) {

			finalEntrada = System.currentTimeMillis() - inicioEntrada;

		}

		/* Set totalUrls */
		numeroUrls = indiceDados.size() / 2;
		int numURL[] = new int[1];
		numURL[0] = numeroUrls;
		int totalNumUrl[] = new int[1];
		MPI.COMM_WORLD.Allreduce(numURL, 0, totalNumUrl, 0, 1, MPI.INT, MPI.SUM);
		totalUrls = totalNumUrl[0];
		double FinalRVT[] = new double[totalUrls];

		if (rank == 0)
			inicioExecucao = System.currentTimeMillis();


		for (int k = 0; k < totalUrls; k++) {
			FinalRVT[k] = 1.0 / totalUrls;
		}

		/* Start the core computation of MPI PageRank */
		mpi_pagerankfunc(dados, indiceDados, numeroUrls, totalUrls, interacoes, limite, FinalRVT);

		if (rank == 0)
			finalExecucao = System.currentTimeMillis() - inicioExecucao;

		/* Save results to a file */
		if (rank == 0) {
			for (int i = 0; i < FinalRVT.length; i++) {
				pageRanks.put(i, FinalRVT[i]);
			}

			inicioSaida = System.currentTimeMillis();
			mpi_write(outfilename, pageRanks);
			finalSaida = System.currentTimeMillis() - inicioSaida;

			System.out.println("Input file : " + args[3]);
			System.out.println("Output file : " + args[4]);
			System.out.println("Number of Iterations: " + args[5]);
			System.out.println("limite: " + args[6]);
			System.out.println("Total I/O time taken:" + (finalEntrada + finalSaida) + " milliseconds");
			System.out.println("Total Computation time taken:" + finalExecucao + " milliseconds");
		}
		/* Release resources e.g. free(adjacency_matrix); */

		MPI.Finalize();

	}

	private static void leituraDados(String filename, HashMap<Integer, ArrayList<Integer>> dados, ArrayList<Integer> indiceDados) throws IOException {

		if(MPI.COMM_WORLD.Rank() == 0) {

			//preenchimento dos nós destino, conforme dispostos no arquivo.
			preencherNosDestino(filename, dados);

			ArrayList<Integer> nos = new ArrayList<>(dados.keySet());
			int totalUrls = dados.size();
			int numeroProcessos = MPI.COMM_WORLD.Size();
			int divisoes = (totalUrls) / numeroProcessos;
			int resto = (totalUrls) % numeroProcessos;
			int strtIndex = 0, j = 0, inicio, h;
			int tamBloco, tamLigacoes;

			for(int i = 0; i < numeroProcessos; i++) {

				h = 0;
				inicio = strtIndex;
				tamBloco = calcularTamanhoBloco(divisoes, resto, i);
				tamLigacoes = tamBloco * 2;

				//vetor de ligações dos nós
				int[] ligacoes = new int[tamLigacoes];

				//repetição para acessar os valores dos nós a cada bloco
				for(strtIndex = inicio; strtIndex <= ((inicio + tamBloco) - 1); strtIndex++) {

					//atribuição ao vetor de ligações, determinando o nó e posteriormente seu número de ligações
					int no = nos.get(j++);
					ligacoes[h++] = no;
					ligacoes[h++] = dados.get(no).size();

				}

				int[] tamDados = new int[1];
				tamDados[0] = tamLigacoes;

				//verificação para o primeiro processo
				if(i == 0) {

					//preenchimento dos valores de ligações a serem usados além do método
					for(int l = 0; l < tamDados[0]; l++) {

						indiceDados.add(ligacoes[l]);

					}

				} else {

					//envio do tamanho dos dados para os demais processos
					MPI.COMM_WORLD.Send(tamDados, 0, 1, MPI.INT, i, 0);
					MPI.COMM_WORLD.Send(ligacoes, 0, tamDados[0], MPI.INT, i, 1);

				}


				for (int k = 0; k < tamBloco * 2; k = k + 2) {
					int source = ligacoes[k];
					int outdegree = ligacoes[k + 1];
					int[] targetList = new int[outdegree];

					ArrayList<Integer> targetUrls = dados.get(source);

					for (int n = 0; n < outdegree; n++) {
						targetList[n] = targetUrls.get(n);
					}

					if (i != 0) {
						MPI.COMM_WORLD.Send(targetList, 0, outdegree, MPI.INT, i, 2);
					}
				}

			}

		} else {

			int[] am_size = new int[1];
			MPI.COMM_WORLD.Recv(am_size, 0, 1, MPI.INT, 0, 0);

			int[] t_indiceDados = new int[am_size[0]];
			MPI.COMM_WORLD.Recv(t_indiceDados, 0, am_size[0], MPI.INT, 0, 1);

			for (int l = 0; l < am_size[0]; l++) {
				indiceDados.add(t_indiceDados[l]);
			}

			int no2Urls = am_size[0];
			for (int p = 0; p < no2Urls; p = p + 2) {
				int sourceUrl = t_indiceDados[p];
				int outdegree = t_indiceDados[p + 1];
				int[] target = new int[outdegree];
				MPI.COMM_WORLD.Recv(target, 0, outdegree, MPI.INT, 0, 2);
				ArrayList<Integer> targetUrls = new ArrayList<>();

				for (int m = 0; m < outdegree; m++) {
					targetUrls.add(target[m]);
				}

				dados.put(sourceUrl, targetUrls);
			}

		}

	}

	//Calcula o tamanho do bloco, para que cada processo assuma este tamanho
	private static int calcularTamanhoBloco(int divisoes, int resto, int i) {

		return resto == 0 ? divisoes : (i < resto) ? (divisoes + 1) : (divisoes);

	}

	private static void preencherNosDestino(String filename, HashMap<Integer, ArrayList<Integer>> dados) throws IOException {

		DataInputStream dataInputStream = new DataInputStream(new FileInputStream("files/" + filename));
		BufferedReader leitorUrls = new BufferedReader(new InputStreamReader(dataInputStream));
		String entrada;

		while((entrada = leitorUrls.readLine()) != null) {

			ArrayList<Integer> urlsDestino = new ArrayList<>();

			//Declaração de array dos nós
			String[] nos = entrada.split(" ");

			//Preenchimento dos nós destino do primeiro nó da linha
			for(int i = 1; i < nos.length; i++) {

				urlsDestino.add(Integer.valueOf(nos[i]));

			}

			//relação dos nós destino com o primeiro nó da linha
			dados.put(Integer.valueOf(nos[0]), urlsDestino);

		}

		dataInputStream.close();

	}

	private static void mpi_pagerankfunc(HashMap<Integer, ArrayList<Integer>> dados, ArrayList<Integer> amIndex, int numeroUrls, int totalUrls,
										 int interacoes, double limite, double[] FinalRVT) {

		/* Definitions of variables */
		double dangling, sum_dangling, intermediate_rank_value, damping_factor = 0.85;
		int source, outdegree, targetUrl, loop = 0;
		ArrayList<Integer> targetUrls;


		/* Allocate memory and initialize values for local_rank_values_table */
		double[] intermediateRV = new double[totalUrls];
		double[] localRV = new double[totalUrls];
		double[] danglingArray = new double[1];
		double[] sumDangling = new double[1];
		double[] deltaArray = new double[1];
		deltaArray[0] = 0.0;

		/* Get MPI rank */
		int rank = MPI.COMM_WORLD.Rank();

		if (rank == 0)
			System.out.println("Max_Iterations: " + interacoes + ", limite: " + limite);
		/* Start computation loop */
		do {
			/* Compute pagerank and dangling values */
			dangling = 0.0;

			for (int i = 0; i < amIndex.size(); i = i + 2) //indiceDados = anIndex
			{
				source = amIndex.get(i);
				targetUrls = dados.get(source);
				outdegree = targetUrls.size();


				for (int j = 0; j < outdegree; j++) {
					targetUrl = targetUrls.get(j);
					intermediate_rank_value = localRV[targetUrl] + FinalRVT[source] / (double) outdegree;
					localRV[targetUrl] = intermediate_rank_value;
				}

				if (outdegree == 0) {
					dangling += FinalRVT[source];
				}
			}


			/* Distribute pagerank values */
			MPI.COMM_WORLD.Allreduce(localRV, 0, FinalRVT, 0, totalUrls, MPI.DOUBLE, MPI.SUM);


			/* Distribute dangling values */
			danglingArray[0] = dangling;
			MPI.COMM_WORLD.Allreduce(danglingArray, 0, sumDangling, 0, 1, MPI.DOUBLE, MPI.SUM);
			sum_dangling = sumDangling[0];

			/* Recalculate the page rank values with damping factor 0.85 */
			/* Root(process 0) computes delta to determine to stop or continue */
			if (rank == 0) {

				double dangling_value_per_page = sum_dangling / totalUrls;

				for (int i = 0; i < totalUrls; i++) {
					FinalRVT[i] = FinalRVT[i] + dangling_value_per_page;
				}

				for (int i = 0; i < totalUrls; i++) {
					FinalRVT[i] = damping_factor * FinalRVT[i] + (1 - damping_factor) * (1.0 / (double) totalUrls);
				}

				deltaArray[0] = 0.0;


				for (int i = 0; i < totalUrls; i++) {

					deltaArray[0] += Math.abs(intermediateRV[i] - FinalRVT[i]);
					intermediateRV[i] = FinalRVT[i];
				}

			}

			MPI.COMM_WORLD.Bcast(deltaArray, 0, 1, MPI.DOUBLE, 0);

			MPI.COMM_WORLD.Bcast(FinalRVT, 0, totalUrls, MPI.DOUBLE, 0);


			for (int k = 0; k < totalUrls; k++) {
				localRV[k] = 0.0;
			}

			if (rank == 0)
				System.out.println("--Current Iteration: " + loop + " delta: " + deltaArray[0]);

		} while (deltaArray[0] > limite && ++loop < interacoes);

	}

	private static void mpi_write(String filename, LinkedHashMap<Integer, Double> sortHash) throws IOException {

		double probabilidades = 0.0;
		int[] keys = new int[sortHash.size()];
		double[] values = new double[sortHash.size()];
		int index = 0;

		for(Double val : sortHash.values()) {

			probabilidades += val;

		}

		for(Map.Entry<Integer, Double> mapEntry : sortHash.entrySet()) {

			keys[index] = Integer.parseInt(mapEntry.getKey().toString());
			values[index] = Double.parseDouble(mapEntry.getValue().toString());
			index++;

		}

		/* Sort the page rank values in ascending order */
		List<Double> pagesRank = new ArrayList<>(sortHash.values());
		Collections.sort(pagesRank);
		ListIterator sorted_page_rank_iterator = pagesRank.listIterator(pagesRank.size());
		int number_web_pages = 0;
		int totalUrls = pagesRank.size();

		/*File operation to write result to output file */
		Writer output = new BufferedWriter(new FileWriter(new File(filename)));
		output.append("\nTop 10 URLs with Highest Page Rank values " + "\n\n" + "--------------------------------------" + "\n");
		output.append("|\t" + "URL" + "\t\t|\t" + "Page Rank" + "\t\t\t|\n" + "--------------------------------------" + "\n");

		while(sorted_page_rank_iterator.hasPrevious() && number_web_pages++ < 10) {

			String str = sorted_page_rank_iterator.previous().toString();
			double pagerankop = Double.valueOf(str);

			/*Get top 10 URLs along with their Page Rank values
			 *and store this list into external output file
			 */

			for(int i = 0; i < totalUrls; i++) {

				if(values[i] == pagerankop) {

					output.write("|\t" + keys[i] + "\t\t|\t" + String.format("%2.17f", pagerankop) + "\t|\n");
					System.out.println("|\t" + keys[i] + "\t\t|\t" + String.format("%2.17f", pagerankop) + "\t|\n");
					output.write("---------------------------------------" + "\n");
					values[i] = -1;

					break;

				}

			}

		}

		output.append("\n" + "Cumulative Sum of Page Rank values\n");
		output.append("--------------------------------------" + "\n");
		output.write(String.format("%1.16f", probabilidades) + "\n");
		output.close();

	}

}