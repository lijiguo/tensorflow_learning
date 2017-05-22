葡萄酒种类识别任务

1、数据描述
	/wine.data，178 * 14的一个double类型的矩阵，第一列表示所属类别，后面13列分别记录当前样本的13个属性。
	
	样本所属类别比例：
		class 1 59
		class 2 71
		class 3 48

2、13个特征分量（属性）
	1) Alcohol
 	2) Malic acid
 	3) Ash
	4) Alcalinity of ash  
 	5) Magnesium
	6) Total phenols
 	7) Flavanoids
 	8) Nonflavanoid phenols
 	9) Proanthocyanins
	10) Color intensity
	11) Hue
	12) OD280/OD315 of diluted wines
	13) Proline

3、任务
	从每个类别的样本集中最多选择80%组成训练集，构建并训练一个至少包含一个隐藏层的神经网络模型；
	并且，将剩下的样本作为测试集，评估模型的分类效果。

4、作业详情
	1) 时间：05/10 00:00 - 05/17 22:00
	2) 提交内容：整个工程项目的压缩包，包含代码、模型、模型说明文档（模型结构，参数设置等），以“学号_姓名_第几次作业”命名。
	3) 提交地址：邮箱ucas_nlp2017@163.com