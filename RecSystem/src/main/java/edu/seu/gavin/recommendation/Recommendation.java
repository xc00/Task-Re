package edu.seu.gavin.recommendation;

import edu.seu.gavin.crawler.Preprocess;
import edu.seu.gavin.crawler.domain.Developer;
import edu.seu.gavin.crawler.domain.HistoryTask;
import edu.seu.gavin.crawler.domain.Task;
import edu.seu.gavin.crawler.service.TaskService;
import edu.seu.gavin.crawler.service.impl.TaskServiceImpl;
import edu.seu.gavin.crawler.util.FileUtil;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.log4j.Logger;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.OutputStreamWriter;
import java.util.*;

/**
 * 整个推荐过程
 */
public class Recommendation {

    private static Logger logger = Logger.getLogger(Recommendation.class);

    public static void main(String[] args) {

        Recommendation recommendation = new Recommendation();
        // 读取A-LSTM模型
        MultiLayerNetwork model = recommendation.loadModel1();
        // 读取XGBoost模型
        Booster booster = recommendation.loadModel2();

        // 对于每一个开发者进行推荐
        Preprocess preprocess = new Preprocess();
        HashMap<Integer, ArrayList<Double>> taskPreferenceMap = preprocess.getTaskPreference();

        String pathDeveloper = System.getProperty("user.dir") + "\\src\\main\\resources\\json\\filter2018-2019";
        ArrayList<String> filePaths = new ArrayList<>();
        FileUtil.getFilePaths(pathDeveloper, filePaths);

        double averagePrecision = 0; // 准确率
        double averageRecall = 0; // 召回率
        double averageF1 = 0; // F1值
        double ndcg = 0; // NDCG值

        int count = 0;
        double maxDcg = 0;

        for(String filePath: filePaths){

            Developer developer = FileUtil.fromFileToDeveloper(filePath);
            // 生成任务序列
            recommendation.getDeveloperTaskPreferenceSequence(developer, taskPreferenceMap);
            // 预测任务特征
            ArrayList<Double> predictTaskPreference = recommendation.predictDeveloperTaskPreference(developer,model);
            // 获得所有待推荐任务
            ArrayList<Integer> candidateTask = recommendation.getAllCandidatedTask(developer);
            // 获得实际参与任务
            ArrayList<Integer> realTasks = recommendation.getAllRealTask(developer, candidateTask);
            // 获得第一阶段推荐任务
            ArrayList<Integer> recommendTask = recommendation.getAllRecommendTask(candidateTask, taskPreferenceMap, predictTaskPreference);
            // 获得第二阶段推荐任务
            ArrayList<Integer> recommendTask2 = recommendation.getAllRecommendTask2(developer,booster,recommendTask,taskPreferenceMap);

            if(candidateTask.size() == 0 || recommendTask.size() == 0 || recommendTask2.size() == 0|| realTasks.size() == 0){
                logger.info("skip");
                continue;

            }

            logger.info("第"+(count+1)+"个开发者");

            // 计算Precison
            double precision = recommendation.computePrecision(recommendTask2,realTasks);
            logger.info("precision: " + precision);
            averagePrecision += precision;
            // 计算Recall
            double recall  = recommendation.computeRecall(recommendTask2,realTasks);
            logger.info("recall: " + recall);
            averageRecall += recall;
            // 计算dcg
            double dcg = recommendation.computeDCG(recommendTask2, developer.getTaskResult());
            if(dcg > maxDcg){
                maxDcg = dcg;
            }
            logger.info("dcg: " + dcg);
            ndcg += dcg;
            count++;
        }

        averagePrecision = averagePrecision / count;
        averageRecall = averageRecall / count;
        averageF1 = recommendation.computeF1(averagePrecision,averageRecall);
        ndcg = ndcg / maxDcg / count;
        logger.info("Precision: " +  averagePrecision);
        logger.info("Recall: " + averageRecall);
        logger.info("F1: " + averageF1);
        logger.info("ndcg: " + ndcg);
    }

    /**
     * 加载模型1（A-LSTM）
     * @return 训练好的A-LSTM模型
     */
    private MultiLayerNetwork loadModel1(){

        String filePath = System.getProperty("user.dir") + "\\src\\main\\resources\\model\\A_LSTM.zip";
        File locationToLoad = new File(filePath);
        MultiLayerNetwork model = null;
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(locationToLoad);
        }catch(Exception e){
            e.printStackTrace();
        }
        return model;
    }

    /**
     *  生成开发者当前偏好序列
     */
    private void getDeveloperTaskPreferenceSequence(Developer developer, HashMap<Integer, ArrayList<Double>> taskPreferenceMap){

        String dataPath = System.getProperty("user.dir") + "\\src\\main\\resources\\data\\";

        Preprocess preprocess = new Preprocess();

        ArrayList<HistoryTask> historyTasks = developer.getTaskResult();
        Collections.sort(historyTasks, new Comparator<HistoryTask>() {
            @Override
            public int compare(HistoryTask o1, HistoryTask o2) {

                return o1.getId() - o2.getId();
            }
        });

        String InputPath = dataPath+"A_LSTM_"+developer.getId()+"_.csv";
        String LabelPath = dataPath+"A_LSTM_"+developer.getId()+"_label.csv";

        try {
            FileOutputStream fos = new FileOutputStream(InputPath);
            OutputStreamWriter osw = new OutputStreamWriter(fos, "GBK");
            FileOutputStream fos2 = new FileOutputStream(LabelPath);
            OutputStreamWriter osw2 = new OutputStreamWriter(fos2, "GBK");

            ArrayList<String> headers = new ArrayList<>();
            headers.addAll(preprocess.getTypeList());
            headers.addAll(preprocess.getSubTypeList());
            headers.addAll(preprocess.getTechnologyList());
            headers.add("cycle");
            headers.add("prize");
            String[] headersStr = headers.toArray(new String[headers.size()]);
            CSVFormat csvFormat = CSVFormat.DEFAULT.withHeader(headersStr);
            csvFormat.withDelimiter(',');
            CSVPrinter csvPrinter = new CSVPrinter(osw, csvFormat);
            CSVPrinter csvPrinter2 = new CSVPrinter(osw2, csvFormat);

            int firstIndex = historyTasks.size() / 2;
            for(int i = firstIndex; i < firstIndex + 5; i++){
                Integer taskId = historyTasks.get(i).getId();
                ArrayList<Double> preference = taskPreferenceMap.get(taskId);
                csvPrinter.printRecord(preference);
            }
            csvPrinter.flush();
            csvPrinter.close();

            csvPrinter2.printRecord(taskPreferenceMap.get(historyTasks.get(firstIndex+5).getId()));
            csvPrinter2.flush();
            csvPrinter2.close();

        }catch(Exception e){
            e.printStackTrace();
        }
    }

    /**
     * 预测当前任务特征
     * @param developer 开发者
     * @param model 模型
     * @return 开发者当前任务偏好特征
     */
    private ArrayList<Double> predictDeveloperTaskPreference(Developer developer, MultiLayerNetwork model){

        ArrayList<Double> predictedDeveloperPreference = new ArrayList<>();
        try {
            String dataPath = System.getProperty("user.dir") + "\\src\\main\\resources\\data\\";
            SequenceRecordReader testFeature = new CSVSequenceRecordReader(1, ",");
            SequenceRecordReader testLabel = new CSVSequenceRecordReader(1, ",");
            testFeature.initialize(new NumberedFileInputSplit(dataPath+"A_LSTM_%d_.csv", developer.getId(), developer.getId()));
            testLabel.initialize(new NumberedFileInputSplit(dataPath+"A_LSTM_%d_label.csv", developer.getId(), developer.getId()));
            DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeature, testLabel, 1, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            while(testData.hasNext()){
                DataSet t = testData.next();
                INDArray features = t.getFeatures();
                INDArray predicted = model.output(features, false);
                double[] predictedPreference = predicted.getRow(0).getColumn(4).toDoubleVector();

                for(int i = 0; i < predictedPreference.length; i++){
                    predictedDeveloperPreference.add(predictedPreference[i]);
                }
            }

        }catch(Exception e){
            e.printStackTrace();
        }
        return  predictedDeveloperPreference;
    }

    /**
     * 获得所有待推荐任务
     * @param developer
     * @return 待推荐任务Id
     */
    private ArrayList<Integer> getAllCandidatedTask(Developer developer){

        ArrayList<Integer> candidatedTasks = new ArrayList<>();

        TaskService taskService = new TaskServiceImpl();
        ArrayList<Task> tasks = taskService.getAllTasks();

        ArrayList<HistoryTask> historyTasks = developer.getTaskResult();
        Collections.sort(historyTasks, new Comparator<HistoryTask>() {
            @Override
            public int compare(HistoryTask o1, HistoryTask o2) {

                return o1.getId() - o2.getId();
            }
        });

        int index = historyTasks.size() / 2;

        HistoryTask lastTask = historyTasks.get(index + 4);
        Date registrationDate = lastTask.getRegistrationDate();
        for(Task task: tasks){
            Date postingDate = task.getPostingDate();
            Date submissionEndDate = task.getSubmissionEndDate();
            if(registrationDate.compareTo(postingDate) >= 0 && registrationDate.compareTo(submissionEndDate) <= 0){
                candidatedTasks.add(task.getId());
            }
        }
        //logger.info("candidate: "+candidatedTasks);
        return candidatedTasks;
    }


    /**
     * 获得第一阶段推荐任务
     */
    private ArrayList<Integer> getAllRecommendTask(ArrayList<Integer> candidateTask, HashMap<Integer, ArrayList<Double>> taskPreferenceMap, ArrayList<Double> predictTaskPreference){

        ArrayList<TaskSim> taskSimArrayList = new ArrayList<>();
        for(Integer id: candidateTask){
            ArrayList<Double> preference = taskPreferenceMap.get(id);
            Preprocess preprocess = new Preprocess();
            double sim = preprocess.computeSimilarity(predictTaskPreference,preference);
            TaskSim taskSim = new TaskSim();
            taskSim.setId(id);
            taskSim.setSim(sim);
            taskSimArrayList.add(taskSim);
        }
        Collections.sort(taskSimArrayList, new Comparator<TaskSim>() {
            @Override
            public int compare(TaskSim o1, TaskSim o2) {
                return o1.sim - o2.sim > 0 ? -1 : 1;
            }
        });

        ArrayList<Integer> taskIds = new ArrayList<>();
        for(int i = 0; i < 50 && i < taskSimArrayList.size(); i++){
            taskIds.add(taskSimArrayList.get(i).getId());
        }
        //logger.info("recom: "+taskIds);
        return taskIds;
    }

    /**
     * 临时类，获得每个任务的相似度
     */
    class TaskSim{
        Integer id;
        double sim;

        public Integer getId() {
            return id;
        }

        public void setId(Integer id) {
            this.id = id;
        }

        public double getSim() {
            return sim;
        }

        public void setSim(double sim) {
            this.sim = sim;
        }

        @Override
        public String toString() {
            return "TaskSim{" +
                    "id=" + id +
                    ", sim=" + sim +
                    '}';
        }
    }

    /**
     * 获得第二阶段推荐任务
     */
    private ArrayList<Integer> getAllRecommendTask2(Developer developer, Booster booster, ArrayList<Integer> recommendTask, HashMap<Integer, ArrayList<Double>> taskPreferenceMap){

        ArrayList<TaskScore> taskScoreArrayList = new ArrayList<>();
        for(Integer id: recommendTask){
            double score = computeScore(developer, booster, id, taskPreferenceMap);
            TaskScore taskScore = new TaskScore();
            taskScore.setId(id);
            taskScore.setScore(score);
            taskScoreArrayList.add(taskScore);
        }
        Collections.sort(taskScoreArrayList, new Comparator<TaskScore>() {
            @Override
            public int compare(TaskScore o1, TaskScore o2) {
                return o1.score- o2.score > 0 ? -1 : 1;
            }
        });
        //logger.info(taskScoreArrayList);

        ArrayList<Integer> taskIds = new ArrayList<>();
        for(int i = 0; i < 20 && i < taskScoreArrayList.size(); i++){
            taskIds.add(taskScoreArrayList.get(i).getId());
        }
        //logger.info("recom: " + taskIds);
        return taskIds;
    }

    /**
     * 临时类，获得每个任务的评分
     */
    class TaskScore{
        Integer id;
        double score;

        public Integer getId() {
            return id;
        }

        public void setId(Integer id) {
            this.id = id;
        }

        public double getScore() {
            return score;
        }

        public void setScore(double score) {
            this.score = score;
        }

        @Override
        public String toString() {
            return "TaskSim{" +
                    "id=" + id +
                    ", score=" + score +
                    '}';
        }
    }

    /**
     * 获得开发者所有实际参与任务
     */
    private ArrayList<Integer> getAllRealTask(Developer developer, ArrayList<Integer> candidateTask){

        Integer id = 0;
        for(Integer candidateId: candidateTask){
            if(candidateId > id){
                id = candidateId;
            }
        }

        ArrayList<HistoryTask> historyTasks = developer.getTaskResult();
        Collections.sort(historyTasks, new Comparator<HistoryTask>() {
            @Override
            public int compare(HistoryTask o1, HistoryTask o2) {

                return o1.getId() - o2.getId();
            }
        });

        int index = historyTasks.size() / 2;

        ArrayList<Integer> realTasks = new ArrayList<>();
        HistoryTask lastTask = historyTasks.get(index + 4);

        for(HistoryTask historyTask: historyTasks){
            if(historyTask.getId() >lastTask.getId() && historyTask.getId() < id){
                realTasks.add(historyTask.getId());
            }
        }
        //logger.info("real: "+ realTasks);
        return realTasks;
    }

    /**
     * 加载模型2（XGBoost模型）
     * @return 训练好的XGBoost模型
     */
    private Booster loadModel2(){

        String filePath = System.getProperty("user.dir") + "\\src\\main\\resources\\model\\XGboost.bin";
        Booster booster = null;
        try {
            booster = XGBoost.loadModel(filePath);
        }catch(Exception e){
            e.printStackTrace();
        }
        return booster;
    }

    /**
     * 计算每个任务的预测评分
     * @param developer
     * @param booster
     * @param recommendTaskId
     * @param taskPreference
     * @return
     */
    private double computeScore(Developer developer, Booster booster, Integer recommendTaskId, HashMap<Integer, ArrayList<Double>> taskPreference){

        // 获得所有任务信息
        TaskService taskService = new TaskServiceImpl();
        ArrayList<Task> tasks = taskService.getAllTasks();
        HashMap<Integer,Task> taskMap = new HashMap<>();
        for(Task task: tasks){
            taskMap.put(task.getId(), task);
        }

        String dataPath = System.getProperty("user.dir") + "\\src\\main\\resources\\data\\XGBoost_test.txt";
        File dataFile = new File(dataPath);
        FileWriter out;
        try {
            out = new FileWriter(dataFile); // 创建文件字符流 写 对象，传递文件对象
            String line = System.getProperty("line.separator");

            int countSimReg = 0; // 相似任务报名次数
            int countSimSub = 0; // 相似任务提交次数
            int countSimWin = 0; // 相似任务获胜次数
            double avgSimCircle = 0;  // 相似任务平均所需开发周期
            double avgSimScore = 0;  // 相似任务平均得分
            int countUnReg = 0; // 未提交任务个数
            double countUnRegCircle = 0; // 未提交任务平均开发周期
            double countUnRegRemainCircle = 0;// 未提交任务平均剩余提交周期
            double countUnRegPrize = 0; // 未提交任务平均报酬
            int regTotal = taskMap.get(recommendTaskId).getNumRegistrants(); // 报名总人数

            ArrayList<HistoryTask> allHistoryTasks = developer.getTaskResult();
            ArrayList<HistoryTask> beforeHistoryTasks = new ArrayList<>();
            for (HistoryTask historyTask : allHistoryTasks) {
                if (historyTask.getId() < recommendTaskId) {
                    beforeHistoryTasks.add(historyTask);
                }
            }

            Preprocess preprocess = new Preprocess();

            for (int j = 0; j < beforeHistoryTasks.size(); j++) {

                HistoryTask beforeHistoryTask = beforeHistoryTasks.get(j);
                Task taskInfo = taskMap.get(beforeHistoryTask.getId());
                if (preprocess.computeSimilarity(taskPreference.get(recommendTaskId), taskPreference.get(beforeHistoryTask.getId())) >= 0.2) {
                    countSimReg++;
                    if (beforeHistoryTask.getSubmissionDate() != null) {
                        countSimSub++;
                        avgSimCircle += ((beforeHistoryTask.getSubmissionDate().getTime() - beforeHistoryTask.getRegistrationDate().getTime()) * 1.0 / (24 * 60 * 60 * 1000));
                    } else {
                        avgSimCircle += ((taskInfo.getSubmissionEndDate().getTime() - beforeHistoryTask.getRegistrationDate().getTime()) * 1.0 / (24 * 60 * 60 * 1000));
                    }
                    if (beforeHistoryTask.getPlacement() <= taskInfo.getPrizes().size()) {
                        countSimWin++;
                    }
                    avgSimScore += beforeHistoryTask.getScore();
                }
                if (beforeHistoryTask.getSubmissionDate() != null) {
                    if (taskInfo.getPostingDate().compareTo(beforeHistoryTask.getRegistrationDate()) >= 0 && taskInfo.getPostingDate().compareTo(beforeHistoryTask.getSubmissionDate()) <= 0) {
                        countUnReg++;
                        countUnRegCircle += ((taskInfo.getSubmissionEndDate().getTime() - taskInfo.getPostingDate().getTime()) * 1.0 / (24 * 60 * 60 * 1000));
                        countUnRegRemainCircle += ((taskInfo.getSubmissionEndDate().getTime() - beforeHistoryTask.getRegistrationDate().getTime()) * 1.0 / (24 * 60 * 60 * 1000));
                        double avgPrize = 0;
                        for (int k = 0; k < taskInfo.getPrizes().size(); k++) {
                            avgPrize += taskInfo.getPrizes().get(k);
                        }
                        avgPrize = avgPrize / taskInfo.getPrizes().size();
                        countUnRegPrize += avgPrize;
                    }
                } else {
                    if (taskInfo.getPostingDate().compareTo(beforeHistoryTask.getRegistrationDate()) >= 0 && taskInfo.getPostingDate().compareTo(taskInfo.getSubmissionEndDate()) <= 0) {
                        countUnReg++;
                        countUnRegCircle += ((taskInfo.getSubmissionEndDate().getTime() - taskInfo.getPostingDate().getTime()) * 1.0 / (24 * 60 * 60 * 1000));
                        countUnRegRemainCircle += ((taskInfo.getSubmissionEndDate().getTime() - beforeHistoryTask.getRegistrationDate().getTime()) * 1.0 / (24 * 60 * 60 * 1000));
                        double avgPrize = 0;
                        for (int k = 0; k < taskInfo.getPrizes().size(); k++) {
                            avgPrize += taskInfo.getPrizes().get(k);
                        }
                        avgPrize = avgPrize / taskInfo.getPrizes().size();
                        countUnRegPrize += avgPrize;
                    }
                }

            }
            if (countSimReg != 0) {
                avgSimCircle = avgSimCircle * 1.0 / countSimReg;
                avgSimScore = avgSimScore * 1.0 / countSimReg;
            } else {
                avgSimCircle = 0;
                avgSimScore = 0;
            }
            if (countUnReg != 0) {
                countUnRegCircle = countUnRegCircle * 1.0 / countUnReg;
                countUnRegRemainCircle = countUnRegRemainCircle * 1.0 / countUnReg;
                countUnRegPrize = countUnRegPrize * 1.0 / countUnReg;
            } else {
                countUnRegCircle = 0;
                countUnRegRemainCircle = 0;
                countUnRegPrize = 0;
            }

            out.write("0 ");
            out.write("1:" + countSimReg + " ");
            out.write("2:" + countSimSub + " ");
            out.write("3:" + countSimWin + " ");
            out.write("4:" + avgSimCircle + " ");
            out.write("5:" + avgSimScore + " ");
            out.write("6:" + countUnReg + " ");
            out.write("7:" + countUnRegCircle + " ");
            out.write("8:" + countUnRegRemainCircle + " ");
            out.write("9:" + countUnRegPrize + " ");
            out.write("10:" + regTotal);
            out.write(line);
            out.flush();
            out.close();
        }catch(Exception e){
            e.printStackTrace();
        }

        double score = 0;
        try{
            String filePath = System.getProperty("user.dir") + "\\src\\main\\resources\\data\\";
            DMatrix testData = new DMatrix(filePath+"XGBoost_test.txt");
            float result[][] = booster.predict(testData);
            score = result[0][0] * 1.0;
        }catch(Exception e){
            e.printStackTrace();
        }

        return score;
    }


    /**
     * 计算准确率
     * @param recom 推荐给开发者的任务
     * @param real 开发者实际参与的任务
     * @return 准确率
     */
    private double computePrecision(ArrayList<Integer> recom, ArrayList<Integer> real){

        int count = 0;
        for(int i = 0; i < real.size(); i++){
            if(recom.contains(real.get(i))){
                count++;
            }
        }
        return count * 1.0 / recom.size();

    }

    /**
     * 计算召回率
     * @param recom 推荐给开发者的任务
     * @param real 实际参与的任务
     * @return 召回率
     */
    private double computeRecall(ArrayList<Integer> recom, ArrayList<Integer> real){

        int count = 0;
        for(int i = 0; i < real.size(); i++){
            if(recom.contains(real.get(i))){
                count++;
            }
        }
        return count * 1.0 / real.size();
    }

    /**
     * 计算F1值
     * @param precision 准确率
     * @param recall 召回率
     */
    private double computeF1(double precision, double recall){

        return 2 * (precision * recall) / (precision + recall);
    }

    /**
     * 计算DCG的值
     * @param recom 推荐的任务
     * @param taskResults 开发者在任务上的结果
     * @return DCG值
     */
    private double computeDCG(ArrayList<Integer> recom, ArrayList<HistoryTask> taskResults){

        double dcg = 0;
        for(int i = 1; i <= recom.size();i++){
            Integer id = recom.get(i-1);
            double score = 0;
            for(HistoryTask taskResult: taskResults){
                Integer taskId = taskResult.getId();
                if(id.equals(taskId)){
                    score = taskResult.getScore();
                    break;
                }
            }
            dcg += ((Math.pow(2,score-1)/(Math.log(i+1))));
        }
        return dcg;
    }
}
