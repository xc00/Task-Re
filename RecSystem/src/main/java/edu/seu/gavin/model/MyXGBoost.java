package edu.seu.gavin.model;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import org.apache.log4j.Logger;

import java.util.HashMap;
import java.util.Map;

/**
 * XGBoost模型
 */
public class MyXGBoost {

    private static Logger logger = Logger.getLogger(MyXGBoost.class);

    private DMatrix trainData = null; //训练数据集

    private static final int TREE_SIZE = 2000; //基学习器个数
    private static final double ETA = 0.5; //学习率
    private static final int MAX_DEPTH = 4; //最大树深
    private static final double  MIN_CHILD_WEIGHT = 0.01; //最小叶子权重
    private static final double SUBSAMPLE = 0.6; //树随机采样比例
    private static final double COLSAMPLE_BYTREE = 0.6; //列随机采样比例

    public static void main(String[] args) {

        MyXGBoost myXGBoost = new MyXGBoost();
        // 加载数据
        logger.info("加载数据");
        myXGBoost.loadData();

        // 构建模型
        logger.info("构建模型");
        myXGBoost.buildModel();

        // 训练并测试模型
        logger.info("训练并测试模型");
        Booster booster = myXGBoost.trainAndTestModel();

        // 保存模型
        logger.info("保存模型");
        myXGBoost.saveModel(booster);
    }

    /**
     *  加载数据
     */
    private void loadData(){
        try{
            String filePath = System.getProperty("user.dir") + "\\src\\main\\resources\\data\\";
            trainData = new DMatrix(filePath+"XGBoost_train.txt");
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    /**
     * 构建模型
     * @return 构建好的模型
     */
    private Map<String, Object> buildModel(){

        Map<String, Object> params = new HashMap<String, Object>() {
            {
                put("eta", ETA);
                put("max_depth", MAX_DEPTH);
                put("min_child_weight", MIN_CHILD_WEIGHT);
                put("subsample", SUBSAMPLE);
                put("colsample_bytree",COLSAMPLE_BYTREE);
                put("objective", "reg:logistic");
                put("eval_metric", "logloss");
            }
        };
        return params;
    }

    /**
     * 对模型进行训练并测试
     */
    private Booster trainAndTestModel(){

        Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {
            {
                put("train", trainData);
            }
        };
        int nround = TREE_SIZE;
        Map<String, Object> params = buildModel();

        Booster booster = null;
        try {
            booster = XGBoost.train(trainData, params, nround, watches, null, null);

        }catch(Exception e){
            e.printStackTrace();
        }

        return booster;

    }

    /**
     * 保存模型
     */
    private void saveModel(Booster booster){

        String filePath = System.getProperty("user.dir") + "\\src\\main\\resources\\model\\XGBoost.bin";
        try {
            booster.saveModel(filePath);
        } catch(Exception e){
            e.printStackTrace();
        }
    }
}
