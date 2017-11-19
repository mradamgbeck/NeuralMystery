package A3NeuralMystery;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class NeuralMystery {
	private class Record {
		private double[] input;
		private double[] output;

		private Record(double[] input, double[] output) {
			this.input = input;
			this.output = output;
		}
	}

	private int numberOfRecords;
	private int numberOfInputs;
	private int numberOfOutputs;
	private double[] outputMinimums;
	private double[] outputMaximums;

	private int numberOfMiddle;
	private int numberOfIterations;
	private double rate;

	private ArrayList<Record> records;

	private double[] input;
	private double[] middle;
	private double[] output;

	private double[] errorMiddle;
	private double[] errorOutput;

	private double[] thetaMiddle;
	private double[] thetaOutput;

	private double[][] matrixMiddle;
	private double[][] matrixOutput;

	public NeuralMystery() {
		numberOfRecords = 0;
		numberOfInputs = 0;
		numberOfOutputs = 0;
		numberOfMiddle = 0;
		numberOfIterations = 0;
		rate = 0;

		records = null;
		input = null;
		middle = null;
		output = null;
		errorMiddle = null;
		errorOutput = null;
		thetaMiddle = null;
		thetaOutput = null;
		matrixMiddle = null;
		matrixOutput = null;

	}

	public void loadTrainingData(String trainingFile) throws IOException {
		Scanner file = new Scanner(new File(trainingFile));

		// read number of records, inputs, and outputs
		numberOfRecords = file.nextInt();
		numberOfInputs = file.nextInt();
		numberOfOutputs = file.nextInt();
		outputMinimums = new double[numberOfOutputs];
		outputMaximums = new double[numberOfOutputs];

		// create empty list of records
		records = new ArrayList<Record>();

		// for each training record
		for (int i = 0; i < numberOfRecords; i++) {
			// read inputs
			double[] input = new double[numberOfInputs];
			for (int j = 0; j < numberOfInputs; j++) {
				input[j] = file.nextDouble();
			}

			// read outputs
			double[] output = new double[numberOfOutputs];
			for (int j = 0; j < numberOfOutputs; j++) {
				output[j] = file.nextDouble();
			}

			// create record and add record to array
			Record record = new Record(input, output);
			records.add(record);
		}
//		preprocessInputs();
//		preprocessOutputs();
		file.close();
	}

	public void setParameters(int numberOfMiddle, int numberOfIterations, int seed, double rate) {
		// set hidden nodes, iterations, and rate
		this.numberOfMiddle = numberOfMiddle;
		this.numberOfIterations = numberOfIterations;
		this.rate = rate;

		// create random gen
		Random random = new Random(seed);

		// create input and output arrays
		input = new double[numberOfInputs];
		middle = new double[numberOfMiddle];
		output = new double[numberOfOutputs];

		// create error arrays
		errorMiddle = new double[numberOfMiddle];
		errorOutput = new double[numberOfOutputs];

		// create theta arrays
		thetaMiddle = new double[numberOfMiddle];
		thetaOutput = new double[numberOfOutputs];

		// initialize thetas at hidden nodes
		for (int i = 0; i < numberOfMiddle; i++) {
			thetaMiddle[i] = 2 * random.nextDouble() - 1;
		}

		// initialize thetas at output nodes
		for (int i = 0; i < numberOfOutputs; i++) {
			thetaOutput[i] = 2 * random.nextDouble() - 1;
		}

		// initialize weights between input and hidden nodes
		matrixMiddle = new double[numberOfInputs][numberOfMiddle];
		for (int i = 0; i < numberOfInputs; i++) {
			for (int j = 0; j < numberOfMiddle; j++) {
				matrixMiddle[i][j] = 2 * random.nextDouble() - 1;
			}
		}

		// initialize weights between hidden and output nodes
		matrixOutput = new double[numberOfMiddle][numberOfOutputs];
		for (int i = 0; i < numberOfInputs; i++) {
			for (int j = 0; j < numberOfMiddle; j++) {
				matrixMiddle[i][j] = 2 * random.nextDouble() - 1;
			}
		}
	}

	public void train() {
		// repeat for the specified number of iterations
		for (int i = 0; i < numberOfIterations; i++) {
			// for each training record
			for (int j = 0; j < numberOfRecords; j++) {
				// calculate inputs and outputs
				forwardCalculation(records.get(j).input);
				// compute errors, update weights and thetas
				backwardCalculation(records.get(j).output);
			}
//			postprocessOutputs();
		}
	}

	private void forwardCalculation(double[] recordInputs) {
		// feed inputs of record
			for (int j = 0; j < numberOfInputs; j++) {
				input[j] = recordInputs[j];
			}
		
		// for each hidden node
		for (int i = 0; i < numberOfMiddle; i++) {
			double sum = 0;

			// compute input at hidden node
			for (int j = 0; j < numberOfInputs; j++) {
				sum += input[j] * matrixMiddle[j][i];
			}

			// add theta
			sum += thetaMiddle[i];

			// compute output at hidden node
			middle[i] = 1 / (1 + Math.exp(-sum));

		}

		// for each output node
		for (int i = 0; i < numberOfOutputs; i++) {
			double sum = 0;

			// compute input at output node
			for (int j = 0; j < numberOfMiddle; j++) {
				sum += middle[j] * matrixOutput[j][i];
			}

			// add theta
			sum += thetaOutput[i];

			// compute output at output node
			output[i] = 1 / (1 + Math.exp(-sum));
		}
	}

	private void backwardCalculation(double[] trainingOutput) {
		// compute error at every output node
		for (int i = 0; i < numberOfOutputs; i++) {
			errorOutput[i] = output[i] * (1 - output[i]) * (trainingOutput[i] - output[i]);
		}

		// computer error at every hidden node
		for (int i = 0; i < numberOfMiddle; i++) {
			double sum = 0;

			for (int j = 0; j < numberOfOutputs; j++) {
				sum += matrixOutput[i][j] * errorOutput[j];
			}

			errorMiddle[i] = middle[i] * (1 - middle[i]) * sum;
		}

		// update weights between hidden and output nodes
		for (int i = 0; i < numberOfMiddle; i++) {
			for (int j = 0; j < numberOfOutputs; j++) {
				matrixOutput[i][j] += rate * middle[i] * errorOutput[j];
			}
		}

		// update weights between input and hidden nodes
		for (int i = 0; i < numberOfInputs; i++) {
			for (int j = 0; j < numberOfMiddle; j++) {
				matrixMiddle[i][j] += rate * input[i] * errorMiddle[j];
			}
		}

		// update thetas at output nodes
		for (int i = 0; i < numberOfOutputs; i++) {
			thetaOutput[i] += rate * errorOutput[i];
		}

		// update thetas at hidden nodes
		for (int i = 0; i < numberOfMiddle; i++) {
			thetaMiddle[i] += rate * errorMiddle[i];
		}
	}

	private ArrayList<Record> test() {
//		preprocessInputs();
//		preprocessOutputs();
		// repeat for the specified number of iterations
				for (int i = 0; i < numberOfIterations; i++) {
					// for each training record
					for (int j = 0; j < numberOfRecords; j++) {
						// calculate inputs and outputs
						forwardCalculation(records.get(j).input);
					}
				}
//				postprocessOutputs();
		// forward pass
		// back pass
		return records;
	}

	// reads from input file and writes to output file
	public void testData(String inputFile, String outputFile) throws IOException {
		Scanner inFile = new Scanner(new File(inputFile));
		PrintWriter outfile = new PrintWriter(new FileWriter(outputFile));

		int numberOfRecords = inFile.nextInt();

		// for each record
		for (int i = 0; i < numberOfRecords; i++) {
			double[] input = new double[numberOfInputs];

			// read input from input file
			for (int j = 0; j < numberOfInputs; j++) {
				input[j] = inFile.nextDouble();
			}

			// find output using the neural net
			ArrayList<Record> testRecords = test();

			for (int j = 0; j < numberOfOutputs; j++) {
				outfile.print(testRecords.get(i).output[j] + " ");
			}
			outfile.println();
		}
		inFile.close();
		outfile.close();
	}

	// reads from input file and writes to output file
	public void validate(String validationFile) throws IOException {
		Scanner scanner = new Scanner(new File(validationFile));

		int numberOfRecords = scanner.nextInt();

		// for each record
		for (int i = 0; i < numberOfRecords; i++) {
			System.out.println("record # " + i);
			// read inputs
			double[] input = new double[numberOfInputs];
			for (int j = 0; j < numberOfInputs; j++) {
				input[j] = scanner.nextDouble();
			}

			// read actual outputs
			double[] actualOutput = new double[numberOfOutputs];
			for (int j = 0; j < numberOfOutputs; j++) {
				actualOutput[j] = scanner.nextDouble();
			}

			// find predicted output
			ArrayList<Record> testRecords = test();
//			postprocessOutputs();

			// write actual and predicted output
			for (int j = 0; j < numberOfOutputs; j++) {
				System.out.println(
						"actual output: " + actualOutput[j] + " predicted output: " + testRecords.get(i).output[j]);
			}
			System.out.println();
		}

		scanner.close();
	}

	private void preprocessInputs() {
		// preprocess inputs
		double minValue = Double.MAX_VALUE;
		double maxValue = 0;
		// find min and max of input column
		for (int i = 0; i < numberOfInputs; i++) {
			for (int j = 0; j < numberOfRecords; j++) {
				if (records.get(j).input[i] < minValue) {
					minValue = records.get(j).input[i];
				}
				if (records.get(j).input[i] > maxValue) {
					maxValue = records.get(j).input[i];
				}
			}
			// scale input to be between 0 and 1
			if (minValue < 0 || maxValue > 1) {
				for (int j = 0; j < numberOfRecords; j++) {

					double oldInput = records.get(j).input[i];
					double newInput = (oldInput - minValue) / (maxValue - minValue);
					records.get(j).input[i] = newInput;
				}
			}
		}
	}

	private void preprocessOutputs() {
		// preprocess outputs
		// find output mins and maximums
		for (int i = 0 ; i < numberOfOutputs; i++) {
			double minValue = Double.MAX_VALUE;
			double maxValue = 0;
			for (int j = 0; j < numberOfRecords; j++) {
				if (records.get(j).output[i] < minValue) {
					minValue = records.get(j).output[i];
				}
				if (records.get(j).output[i] > maxValue) {
					maxValue = records.get(j).output[i];
				}
			}
			// set global minimums and maximums
			outputMinimums[i] = minValue;
			outputMaximums[i] = maxValue;

			// scale output to be between 0 and 1
			if (minValue < 0 || maxValue > 1) {
				for (int j = 0; j < numberOfRecords; j++) {

					double oldOutput = records.get(j).output[i];
					double newOutput = (oldOutput - minValue) / (maxValue - minValue);
					records.get(j).output[i] = newOutput;
				}
			}
		}
	}

	private ArrayList<Record> postprocessOutputs() {
		// postprocess outputs
		for (int i = 0; i < numberOfOutputs; i++) {
			for (int j = 0; j < numberOfRecords; j++) {
				// scale output to its proper size
				records.get(j).output[i] = (records.get(j).output[i] * (outputMaximums[i] - outputMinimums[i])
						+ outputMinimums[i]);
			}
		}
		return records;
	}
}
