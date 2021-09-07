一、需要的软件（包括其环境配置）：
1、IntelliJ IDEA

2、Maven

3、git

4、jdk

二、除项目pom文件中以外需要的依赖

1、dl4j_attention
   单独在maven中安装
   安装方式：在dl4j_attention文件夹下打开命令行执行mvn clean install

2、xgboost4j
   将xgboost4j文件夹下内容单独加到项目的依赖jar下(External Libraries)

三、项目结构

1、java目录

   crawler包：用于爬取数据的包

   model包：2个模型（A-LSTM和XGBoost）

   recommendation包：用于推荐

2、resources目录

   filter2018-2019文件夹:2018-2019年的开发者信息

   filterTasks2018-2019文件：2018-2019年的任务信息

   status文件：所有任务的状态

   subTypes文件：所有任务的子类型

   technologies文件：所有任务的技能