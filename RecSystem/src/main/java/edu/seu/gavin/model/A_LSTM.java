package edu.seu.gavin.model;

import org.apache.log4j.Logger;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import tech.dubs.dl4j.contrib.attention.conf.RecurrentAttentionLayer;

import java.io.File;

/**
 * 基于注意力机制的LSTM
 */
public class A_LSTM {

    private static Logger logger = Logger.getLogger(A_LSTM.class);

    private DataSetIterator trainData = null; //训练数据集
    private DataSetIterator testData = null; //测试数据集

    private static final int INPUT_SIZE = 173; //输入层节点数
    private static final int HIDDEN_SIZE = 128; //隐藏层节点数
    private static final int OUTPUT_SIZE = 173; //输出层节点数
    private static final double LEARNING_RATE = 0.01; //学习率
    private static final double DROPOUT = 0.6; //Drop概率
    private static final int EPOCHS = 100; //迭代次数

    public static void main(String[] args) {

        A_LSTM a_lstm = new A_LSTM();
        // 加载数据
        logger.info("加载数据");
        a_lstm.loadData();
        // 构建模型
        logger.info("构建模型");
        MultiLayerNetwork model = a_lstm.buildModel();
        model.init();
        //model.setLearningRate(LEARNING_RATE);
        logger.info("训练并测试模型");
        // 训练并测试模型
        a_lstm.trainAndTestModel(model);
        logger.info("保存模型");
        a_lstm.saveModel(model);

    }

    /**
     *  加载数据
     */
    private void loadData(){
        try{
            String filePath = System.getProperty("user.dir") + "\\src\\main\\resources\\data\\";

            SequenceRecordReader trainFeature = new CSVSequenceRecordReader(1, ",");
            SequenceRecordReader trainLabel = new CSVSequenceRecordReader(1, ",");
            trainFeature.initialize(new NumberedFileInputSplit(filePath+"A_LSTM_Input_%d.csv", 1, 30000));
            trainLabel.initialize(new NumberedFileInputSplit(filePath+"A_LSTM_Label_%d.csv", 1, 30000));
            trainData = new SequenceRecordReaderDataSetIterator(trainFeature, trainLabel, 10, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            SequenceRecordReader testFeature = new CSVSequenceRecordReader(1, ",");
            SequenceRecordReader testLabel = new CSVSequenceRecordReader(1, ",");
            testFeature.initialize(new NumberedFileInputSplit(filePath+"A_LSTM_Input_%d.csv", 4001, 5000));
            testLabel.initialize(new NumberedFileInputSplit(filePath+"A_LSTM_Label_%d.csv", 4001, 5000));
            testData = new SequenceRecordReaderDataSetIterator(testFeature, testLabel, 10, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        }catch(Exception e){
            e.printStackTrace();
        }
    }

    /**
     * 构建模型
     * @return 构建好的模型
     */
    private MultiLayerNetwork buildModel(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                //.dropOut(DROPOUT)
                .list()
                .layer(0, new LSTM.Builder().nOut(HIDDEN_SIZE).activation(Activation.TANH)
                        .build())
                .layer(1, new RecurrentAttentionLayer.Builder().nOut(HIDDEN_SIZE)
                       .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(Activation.SIGMOID)
                        .nOut(OUTPUT_SIZE).build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.recurrent(INPUT_SIZE))
                .pretrain(false).backprop(true)
                .build();
        return new MultiLayerNetwork(conf);
    }

    /**
     * 对模型进行训练并测试
     * @param model 模型
     */
    private void trainAndTestModel(MultiLayerNetwork model){
        model.setListeners(new ScoreIterationListener(10));
        for(int i = 1; i <= EPOCHS; i++){
            model.fit(trainData);
            trainData.reset();
            logger.info("第"+i+"次迭代结束");
            RegressionEvaluation evaluation = new RegressionEvaluation(173);

            while(testData.hasNext()){
                DataSet t = testData.next();
                INDArray features = t.getFeatures();
                INDArray labels = t.getLabels();
                INDArray predicted = model.output(features,true);
                evaluation.evalTimeSeries(labels,predicted);
            }
            logger.info("测试集结果："+ evaluation.averageMeanSquaredError());
            testData.reset();
        }
    }

    /**
     * 保存模型
     * @param model 模型
     */
    private void saveModel(MultiLayerNetwork model){

        String filePath = System.getProperty("user.dir") + "\\src\\main\\resources\\model\\A_LSTM.zip";
        File locationToSave = new File(filePath);

        try {
            ModelSerializer.writeModel(model, locationToSave, false);
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}
