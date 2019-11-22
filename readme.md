https://github.com/gujingxiao/Lane-Segmentation-Solution-For-BaiduAI-Autonomous-Driving-Competition

https://blog.csdn.net/wangdongwei0/article/details/89093095



比赛用数据集, 含三个包, 包内各含图像文件(但包内标注文件已失效). 标注数据独立有一个包, 解压后, 标注数据的路径和图像文件可以一一对应. 

Training 数据介绍：

1: The mapping from the labeled Id to training Id is defined as below. We borrow the format from Cityscape.

Label = namedtuple( 'Label' , ['name' ,'id','trainId','category','categoryId','hasInstances','ignoreInEval','color'] )
labels = [
# name id trainId category catId hasInstances ignoreInEval color
Label( 'void' , 0 , 0, 'void' , 0 , False , False , ( 0, 0, 0) ),
Label( 's_w_d' , 200 , 1 , 'dividing' , 1 , False , False , ( 70, 130, 180) ),
Label( 's_y_d' , 204 , 1 , 'dividing' , 1 , False , False , (220, 20, 60) ),
Label( 'ds_w_dn' , 213 , 1 , 'dividing' , 1 , False , True , (128, 0, 128) ),
Label( 'ds_y_dn' , 209 , 1 , 'dividing' , 1 , False , False , (255, 0, 0) ),
Label( 'sb_w_do' , 206 , 1 , 'dividing' , 1 , False , True , ( 0, 0, 60) ),
Label( 'sb_y_do' , 207 , 1 , 'dividing' , 1 , False , True , ( 0, 60, 100) ),
Label( 'b_w_g' , 201 , 2 , 'guiding' , 2 , False , False , ( 0, 0, 142) ),
Label( 'b_y_g' , 203 , 2 , 'guiding' , 2 , False , False , (119, 11, 32) ),
Label( 'db_w_g' , 211 , 2 , 'guiding' , 2 , False , True , (244, 35, 232) ),
Label( 'db_y_g' , 208 , 2 , 'guiding' , 2 , False , True , ( 0, 0, 160) ),
Label( 'db_w_s' , 216 , 3 , 'stopping' , 3 , False , True , (153, 153, 153) ),
Label( 's_w_s' , 217 , 3 , 'stopping' , 3 , False , False , (220, 220, 0) ),
Label( 'ds_w_s' , 215 , 3 , 'stopping' , 3 , False , True , (250, 170, 30) ),
Label( 's_w_c' , 218 , 4 , 'chevron' , 4 , False , True , (102, 102, 156) ),
Label( 's_y_c' , 219 , 4 , 'chevron' , 4 , False , True , (128, 0, 0) ),
Label( 's_w_p' , 210 , 5 , 'parking' , 5 , False , False , (128, 64, 128) ),
Label( 's_n_p' , 232 , 5 , 'parking' , 5 , False , True , (238, 232, 170) ),
Label( 'c_wy_z' , 214 , 6 , 'zebra' , 6 , False , False , (190, 153, 153) ),
Label( 'a_w_u' , 202 , 7 , 'thru/turn' , 7 , False , True , ( 0, 0, 230) ),
Label( 'a_w_t' , 220 , 7 , 'thru/turn' , 7 , False , False , (128, 128, 0) ),
Label( 'a_w_tl' , 221 , 7 , 'thru/turn' , 7 , False , False , (128, 78, 160) ),
Label( 'a_w_tr' , 222 , 7 , 'thru/turn' , 7 , False , False , (150, 100, 100) ),
Label( 'a_w_tlr' , 231 , 7 , 'thru/turn' , 7 , False , True , (255, 165, 0) ),
Label( 'a_w_l' , 224 , 7 , 'thru/turn' , 7 , False , False , (180, 165, 180) ),
Label( 'a_w_r' , 225 , 7 , 'thru/turn' , 7 , False , False , (107, 142, 35) ),
Label( 'a_w_lr' , 226 , 7 , 'thru/turn' , 7 , False , False , (201, 255, 229) ),
Label( 'a_n_lu' , 230 , 7 , 'thru/turn' , 7 , False , True , (0, 191, 255) ),
Label( 'a_w_tu' , 228 , 7 , 'thru/turn' , 7 , False , True , ( 51, 255, 51) ),
Label( 'a_w_m' , 229 , 7 , 'thru/turn' , 7 , False , True , (250, 128, 114) ),
Label( 'a_y_t' , 233 , 7 , 'thru/turn' , 7 , False , True , (127, 255, 0) ),
Label( 'b_n_sr' , 205 , 8 , 'reduction' , 8 , False , False , (255, 128, 0) ),
Label( 'd_wy_za' , 212 , 8 , 'attention' , 8 , False , True , ( 0, 255, 255) ),
Label( 'r_wy_np' , 227 , 8 , 'no parking' , 8 , False , False , (178, 132, 190) ),
Label( 'vom_wy_n' , 223 , 8 , 'others' , 8 , False , True , (128, 128, 64) ),
Label( 'om_n_n' , 250 , 8 , 'others' , 8 , False , False , (102, 0, 204) ),
Label( 'noise' , 249 , 0 , 'ignored' , 0 , False , True , ( 0, 153, 153) ),
Label( 'ignored' , 255 , 0 , 'ignored' , 0 , False , True , (255, 255, 255) ),
]
1): "name": 代表此类别的名字，其中，"s" 代表singe， "d" 代表double"w", 代表white， "y" 代表yellow，
2): "id":代表的是标注时候的Id,也是现在发布数据中的class Id。
3): "trainId": 代表的是建议的training Id。 在进行训练的过程中，数据的classId要是连续的整数。所以需要把label图像中的标注Id转化为trainId进行训练。
4): "category" 代表的是不同的类别的具体含义；
5): "catId" 代表的不同的不同的类别的Id，在这里我们的training Id 与 catId相同；
6): "hasInstances", 代表这个类别是不是示例级别，这个在此task中并无实际意义。主要是与cityscape的mapping format保持一致；
7): "ignoreInEval", 代表这个类别在评估时，是否忽略。请注意，不是所有的类别都在最终的结果中进行评估。原因是，有一些类别在此数据集中并不存在；
8): "color", 代表不同的class对应的color map；

2： 此数据的标注是采用在三维点云上标注，然后投影到2D图像的策略。所以会出现，在2D图像上很小的车道线也被标注出来的情况。

3: 数据提交格式如下:
1): 提交的预测结果图要与我们提供的label图像名字与格式保持完全一致，否则上传无法通过格式检查；
2): 务必将预测结果图像中的预测Id (如，0,1,2,3,4,5,6,7,8)，转化为标注Id，因为我们答案存储的是以标注Id的形式。如预测值为1的label要转化为200，204，206， 207， 209，或者213都可以，
因为在评估的时候，这几个值是看作同一类别的，都会判断为正确。

Contact:
If you have any problem of the dataset, please feel free to send us an email to apollo-scape@baidu.com or send a message via qq group: 百度AI Studio 518588005

 