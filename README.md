# Knowledge_Graph_begin
知识图谱--Relation Classification
## dataSet
 * entity2id.txt 
   
   实物及对应索引，共14951个实物

 * relation2id.txt
 	
   关系及对应索引，共1345个关系

 * train.txt
 	
   训练集，前两列为实物，第三列为关系。 shape:(483142,3)

 * test.txt
 	
   测试集，前两列为实物，第三列为关系。 shape:(59071,3)

 * valid.txt
 	
   验证集，前两列为实物，第三列为关系。 shape:(50000,3)
   
## Task
* 对知识图谱里的关系分类
* 评价指标用AUC score，由于数据已分隔好，无需使用n-fold验证。
* Note：需要对知识图谱有一定的了解，其次把其作为分类问题来做，注意这是多分类问题，可考虑先对知识图谱进行处理学习，然后再分类。
* 参考文献: [Knowledge Graph Embedding via Dynamic Mapping Matrix](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Knowledge%20Graph%20Embedding%20via%20Dynamic%20Mapping%20Matrix.pdf)

## result
![image](https://github.com/Aplicity/Knowledge_Graph_begin/blob/master/images/Figure_1.png)
