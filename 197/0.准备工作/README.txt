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
说明：只需要主动运行“生成this_dataset.py”文件，生成“this_dataset.py”用于后续问题的计算。
其他的 .py文件都是被调用的，不需要主动运行。
 “this_dataset.bak”、“this_dataset.dat”、“this_dataset.dir” 是shelve生成的数据存储文件，不需要主动运行。
 “__pycache__” 文件夹中的文件是Python在import过程中自动生成的，不需要主动运行。