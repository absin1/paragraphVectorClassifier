package dl4j_examples.paragraphvectors;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.util.UUID;

public class CreateData {
	private static int c = 0;

	public static void main(String[] args) {
		// Test();
		Train();
		System.exit(0);
	}

	private static void Train() {
		try {
			(new CreateData()).getTrainData();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private static void Test() {
		try {
			(new CreateData()).getTestData();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private void getTestData() throws IOException {
		File doc = new File("datasrc/test.txt");
		Reader in = new FileReader(doc);
		BufferedReader bufferedReader = new BufferedReader(in);
		String text = "";
		String line;
		while ((line = bufferedReader.readLine()) != null) {
			c++;
			text += line;
			String[] split2 = line.split(" ");
			UUID uuid = UUID.randomUUID();
			File folder = new File("C:\\Users\\Unimax\\git\\paragraphVectorClassifier\\datasrc\\unlabelled\\"
					+ split2[0].trim().replaceAll(":", "_") + "_" + uuid.toString());
			if (!folder.exists())
				folder.mkdirs();
			File file = new File("C:\\Users\\Unimax\\git\\paragraphVectorClassifier\\datasrc\\unlabelled\\"
					+ split2[0].trim().replaceAll(":", "_") + "_" + uuid.toString() + "\\"
					+ split2[0].trim().replaceAll(":", "_") + "_" + uuid.toString() + ".txt");
			file.createNewFile();
			PrintWriter printWriter = new PrintWriter(file);
			String question = "";
			for (int i = 1; i < split2.length; i++) {
				question += split2[i] + " ";
			}
			printWriter.print(question.trim());
			printWriter.close();
		}
		System.out.println(c);
		bufferedReader.close();

	}

	private void getTrainData() throws IOException {
		File doc = new File("datasrc/LabelledData.txt");
		Reader in = new FileReader(doc);
		BufferedReader bufferedReader = new BufferedReader(in);
		String text = "";
		String line;
		while ((line = bufferedReader.readLine()) != null) {
			c++;
			text += line;
			String[] split2 = line.split(",,,");
			UUID uuid = UUID.randomUUID();
			File folder = new File(
					"C:\\Users\\Unimax\\git\\paragraphVectorClassifier\\datasrc\\labelled\\" + split2[1].trim());
			if (!folder.exists())
				folder.mkdirs();
			File file = new File("C:\\Users\\Unimax\\git\\paragraphVectorClassifier\\datasrc\\labelled\\"
					+ split2[1].trim() + "\\" + uuid.toString() + ".txt");
			file.createNewFile();
			PrintWriter printWriter = new PrintWriter(file);
			String question = split2[0].trim();
			printWriter.print(question.trim());
			printWriter.close();
		}
		System.out.println(c);
		bufferedReader.close();

	}
}
