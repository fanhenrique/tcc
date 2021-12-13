'''
Creates falese/fakes matrices for RNA input

use: python3 <adj-size> <time-series-size>
'''

import sys
import random
import numpy
# with open('/home/fanhenrique/Google Drive/tcc/tccTraces/out_matrices/adj.csv') as file:
# 	for i in range(int(sys.argv[1])):
# 		for j in range(int(sys.argv[1])):


# with open('out/out_matrices/monitoring_weigths.csv', 'w') as file:
# 	for i in range(int(int(sys.argv[1]))):
# 		for j in range(int(sys.argv[2])-1):
# 			file.write('100,')
# 		file.write('100\n')


a = numpy.random.randint(2, size = (int(sys.argv[1]), int(sys.argv[1])))

print(a)

print(a.shape)

with open('../out/out-matrices/monitoring-adj.csv', 'w') as file:
	for i in range(a.shape[0]):
		for j in range(a.shape[1]-1):
			file.write(str(a[i,j])+',')				
		file.write(str(a[i, a.shape[1]-1])+'\n')	



# with open('out/out_matrices/monitoring_weigths.csv', 'w') as file:
# 	for i in range(int(int(sys.argv[2]))):
# 		for j in range(int(sys.argv[1])-1):
# 			file.write(str(random.uniform(20.0,120.0))+',')
# 		file.write(str(random.uniform(20.0,120.0))+'\n')



with open('../out/out-matrices/monitoring-weigths.csv', 'w') as file:
	for i in range(int(int(sys.argv[2]))):
		for j in range(int(sys.argv[1])-1):
			file.write(str(random.randint(98,100))+',')
		file.write(str(random.randint(98,100))+'\n')



# with open('out/out_matrices/monitoring_weigths.csv', 'w') as file:
# 	for i in range(int(int(sys.argv[1])/2)):
# 		for j in range(int(sys.argv[2])-1):
# 			file.write(str(random.randint(20,150))+',')
# 		file.write(str(random.randint(20,150))+'\n')

# 	for i in range(int(int(sys.argv[1])/2)+1):
# 		for j in range(int(sys.argv[2])-1):
# 			file.write(str(random.randint(100,300))+',')
# 		file.write(str(random.randint(100,300))+'\n')


# with open('out/out_matrices/monitoring_weigths.csv', 'w') as file:
# 	for i in range(int(int(sys.argv[1])/4)):
# 		for j in range(int(sys.argv[2])-1):
# 			file.write(str(random.randint(20,40))+',')
# 		file.write(str(random.randint(20,40))+'\n')

# 	for i in range(int(int(sys.argv[1])/4)):
# 		for j in range(int(sys.argv[2])-1):
# 			file.write(str(random.randint(2900,3000))+',')
# 		file.write(str(random.randint(2900,3000))+'\n')

# 	for i in range(int(int(sys.argv[1])/4)):
# 		for j in range(int(sys.argv[2])-1):
# 			file.write(str(random.randint(20,40))+',')
# 		file.write(str(random.randint(20,40))+'\n')

# 	for i in range(int(int(sys.argv[1])/4)):
# 		for j in range(int(sys.argv[2])-1):
# 			file.write(str(random.randint(2900,3000))+',')
# 		file.write(str(random.randint(2900,3000))+'\n')


print('update: out/out_matrices/monitoring-adj.csv')
print('update: out/out_matrices/monitoring-weigths.csv')