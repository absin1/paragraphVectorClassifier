package org.deeplearning4j.examples.nlp.paragraphvectors;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.List;

import org.deeplearning4j.examples.nlp.paragraphvectors.tools.LabelSeeker;
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.MeansBuilder;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is basic example for documents classification done with DL4j
 * ParagraphVectors. The overall idea is to use ParagraphVectors in the same way
 * we use LDA: topic space modelling.
 *
 * In this example we assume we have few labeled categories that we can use for
 * training, and few unlabeled documents. And our goal is to determine, which
 * category these unlabeled documents fall into
 *
 *
 * Please note: This example could be improved by using learning cascade for
 * higher accuracy, but that's beyond basic example paradigm.
 *
 * @author raver119@gmail.com
 */
public class ParagraphVectorsClassifierExample {

	ParagraphVectors paragraphVectors;
	LabelAwareIterator iterator;
	TokenizerFactory tokenizerFactory;

	private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsClassifierExample.class);

	public static void main(String[] args) throws Exception {
		Long now = System.currentTimeMillis();
		ParagraphVectorsClassifierExample app = new ParagraphVectorsClassifierExample();
		app.makeParagraphVectors(now);
		System.out.println("Starting unlabeled data after  -> " + (System.currentTimeMillis() - now) + " milliseconds");
		app.checkUnlabeledData();
		System.out.println("Exiting after  -> " + (System.currentTimeMillis() - now) + " milliseconds");
		System.exit(0);
		/*
		 * Your output should be like this:
		 * 
		 * Document '[ABBR_abb_60c29f33-373d-41e3-9295-d670f0156a23]>>What is the
		 * abbreviation for micro ?' falls into the following categories: affirmation:
		 * -0.09517832100391388 unknown: -0.30406102538108826 what: 0.5202820897102356
		 * when: -0.12125464528799057 who: -0.1165885478258133
		 * 
		 * so,now we know categories for yet unseen documents
		 */
	}

	void makeParagraphVectors(Long now) throws Exception {
		iterator = new FileLabelAwareIterator.Builder().addSourceFolder(new File("datasrc/labelled")).build();

		System.out.println("Training files added after ->" + (System.currentTimeMillis() - now) + "milliseconds");

		tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
		System.out.println("Training files tokenized after ->" + (System.currentTimeMillis() - now) + "milliseconds");

		// ParagraphVectors training configuration
		paragraphVectors = new ParagraphVectors.Builder().learningRate(0.025).minLearningRate(0.001).batchSize(1000)
				.epochs(20).iterate(iterator).trainWordVectors(true).tokenizerFactory(tokenizerFactory).build();
		System.out.println("Paragraph vectors created after ->" + (System.currentTimeMillis() - now) / 1000);

		// Start model training
		paragraphVectors.fit();

		System.out.println("Paragraph2vec trained after ->" + (System.currentTimeMillis() - now) + "milliseconds");

	}

	void checkUnlabeledData() throws IOException {
		/*
		 * At this point we assume that we have model built and we can check which
		 * categories our unlabeled document falls into. So we'll start loading our
		 * unlabeled documents and checking them
		 */
		FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
				.addSourceFolder(new File("datasrc/unlabelled")).build();
		/*
		 * Now we'll iterate over unlabeled data, and check which label it could be
		 * assigned to Please note: for many domains it's normal to have 1 document fall
		 * into few labels at once, with different "weight" for each.
		 */
		MeansBuilder meansBuilder = new MeansBuilder((InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(),
				tokenizerFactory);
		LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
				(InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

		while (unClassifiedIterator.hasNextDocument()) {
			LabelledDocument document = unClassifiedIterator.nextDocument();
			INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
			List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

			/*
			 * please note, document.getLabel() is used just to show which document we're
			 * looking at now, as a substitute for printing out the whole document name. So,
			 * labels on these two documents are used like titles, just to visualize our
			 * classification done properly
			 */
			String replaceAll = document.getLabels().get(0).toString().replace("[", "").replace("]", "");

			File file = new File("datasrc/unlabelled" + replaceAll + "/" + replaceAll + ".txt");
			Reader in = new FileReader(file);
			BufferedReader bufferedReader = new BufferedReader(in);
			String question = "";
			String line = "";
			while ((line = bufferedReader.readLine()) != null)
				question += line;
			bufferedReader.close();
			System.out.println(
					"Document '" + document.getLabels() + ">>" + question + "' falls into the following categories: ");
			for (Pair<String, Double> score : scores) {
				System.out.println("        " + score.getFirst() + ": " + score.getSecond());
			}
		}

	}
}
