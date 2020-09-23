### EdgeDetection
### 入口函数为main/main.py
- main 目录
    - main2.py ： 入口函数，一般用到的函数汇总
-   matrix目录
    - check_num : 对 3 * 3 矩阵进行连边操作
    - check_mat.py 
        '''
        class功能：只对外调用bsmall_v2(gray)->markMat,show_edge(martMat)->bin_img
        传入一个原始矩阵，经过连边，大小边判断，矛盾点判断，可能的边缘信息判断，返回一个边缘点矩阵
        '''
       
    - dis_degree.py : 求区分度算法，3x3矩阵分边，爬梯子算法
    - merge_region.py : 
        '''
        如果同一区域的最大区分度 大于 不同区域的的最大差，则代表是同一区域，合并
        同一区域的最大区分度：同一区域max - 同一区域min
        不同区域的最大差：大区域的最小值 - 小区域的最大值
        '''
    - reduce_md.py : 暂时没用
   
 - measure_method 目录
    -  measure.py ：评测指标
 - repair_md 目录
    - repair_md.py ： 修复矛盾点
 - repair_noise 目录
    - repair_noise.py : 修复噪声点
 
 - image 目录 
    -  测试图片
 - read_image 目录
    - image_path.py: 图片路径函数
    - img_operator.py: 读取图片的相关函数封装
 - tools 目录
    - tools.py : 将矩阵输出到excel表格，输出到直方图等相关函数