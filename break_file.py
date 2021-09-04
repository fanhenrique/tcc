import sys

with open(sys.argv[1], 'r') as ifile:

	v = []

	for line in ifile:
		v.append(line)

with open(sys.argv[2], 'w') as ofile:
	
	for l in v[int(sys.argv[3]):int(sys.argv[4])]:
		ofile.write(l)

