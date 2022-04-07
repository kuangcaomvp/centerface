环境安装：
	
	pip install tensorflow-gpu==1.14 //没有gpu  pip install tensorflow==1.14 cpu下跑测试可以
	pip install tqdm
	pip isntall opencv-python
	pip install matplotlib
	pip install easydict

//checkpoints下是在celebaA数据集训练的人脸检测和关键点检测模型 可以直接跑测试
测试图片 python image_demo_hrnet.py
测试摄像头 python video_demo_hrnet.py

训练 ：

	数据格式见2.txt 和 test.txt
	格式  图片路径+'\t' + class_id,reid,(x,y,w,h)相对与图片长宽的比值,landmarks坐标相对与图片长宽的比值
	core/config.py  配置文件  简单一看就懂
	python test.py可以看看数据集准备是否正确
	python train.py 训练


模型冻结：

	python freeze_graph_hrnet.py

