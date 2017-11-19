package A3NeuralMystery;

import java.io.IOException;

public class NeuralMysteryTester {
	public static void main(String[] args) throws IOException {
		// create network
		NeuralMystery network = new NeuralMystery();

		// load training data
		network.loadTrainingData(
				"C:\\Users\\a\\Desktop\\Class\\COSC461 Heuristics\\src\\A3NeuralMystery\\mystery_train.txt");

		// set params of network
		network.setParameters(6, 200, 12345, 0.05);

		// train network
		network.train();
		
		network.validate("C:\\Users\\a\\Desktop\\Class\\COSC461 Heuristics\\src\\A3NeuralMystery\\mystery_validate");

		// test network
		network.testData("C:\\Users\\a\\Desktop\\Class\\COSC461 Heuristics\\src\\A3NeuralMystery\\mystery_test.txt",
				"C:\\Users\\a\\Desktop\\Class\\COSC461 Heuristics\\src\\A3NeuralMystery\\mystery_out.txt");
	}
}
