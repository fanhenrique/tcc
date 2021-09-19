import sys
import random
# with open('/home/fanhenrique/Google Drive/tcc/tccTraces/out_matrices/adj.csv') as file:
# 	for i in range(int(sys.argv[1])):
# 		for j in range(int(sys.argv[1])):


with open('out/out_matrices/monitoring_weigths.csv', 'w') as file:
	for i in range(int(int(sys.argv[1]))):
		for j in range(int(sys.argv[2])-1):
			file.write(str(random.randint(99,100))+',')
		file.write(str(random.randint(99,100))+'\n')


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

