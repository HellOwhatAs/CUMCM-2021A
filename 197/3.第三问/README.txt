运行要求：
Python 3.8.10 64-bit
	pandas                    1.2.0
	numpy                     1.20.3+mkl
	numba                     0.54.0
	matplotlib                3.3.3
	matplotlib-inline         0.1.2
	sympy                     1.7.1
	scipy                     1.6.0
	openpyxl                  3.0.7
Jupyter notebook（用于运行 .ipynb文件）
-----------------------------------------------
说明：运行 “main.py” 会打印迭代中的均方根误差，最后打印出工作抛物面的接收率。
运行 “计算球面接收率.py” 会打印出球面的接收率。
运行 “反射面板是平面的情况的接收率.py” 会打印出将反射面板当作平面的情况下迭代中的均方根误差，最后打印出由平面组成的工作抛物面的接收率。
 “this_dataset.bak”、“this_dataset.dat”、“this_dataset.dir” 是shelve生成的数据存储文件，不需要主动运行。