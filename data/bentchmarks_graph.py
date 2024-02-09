import matplotlib.pyplot as plt

data = [
	[(4,4),		16.109],
	[(4,8),		10.014],
	[(8,8),		8.4657],
	[(8,12),	9.0153],
	[(8,16),	8.8721],
	[(12,12),	9.5125],
	[(12,16),	8.8382],
	[(16,16),	8.5458],
	[(16,24),	8.7968],
	[(24,24),	11.190],
	[(24,32),	9.6666],
	[(32,32),	8.2290],
]

x = ['x'.join(map(str,data[i][0])) for i in range(len(data))]
y = [data[i][1] for i in range(len(data))]

x_indices = range(len(x))

plt.plot(x_indices, y, marker='o', linestyle='-')
plt.xticks(x_indices, x)

plt.xlabel('Block Size')
plt.ylabel('Time (s)')
plt.title('Block Bentchmarks')
plt.show()


data = [
#	["CPU",			29*60+	29.177],
#	["cuda",		1*60+	11.783],
	["vtable",				3.2315],
	["sh mem",				2.7473],
	["fast_math",			2.0407],
	["stack size",			1.8742],
	["host mem",			1.6197],
#	["dynamic",		3*60+	30.874],
	["sh samples",			1.5318],
	["32x32 blocks",		1.5152],
]


x = [data[i][0] for i in range(len(data))]
#x = [i for i in range(len(data))]
y = [data[i][1] for i in range(len(data))]

x_indices = range(len(x))

plt.plot(x_indices, y, marker='o', linestyle='-')
plt.xticks(x_indices, x)

plt.xlabel('Optimization')
plt.ylabel('Time (s)')
plt.title('Time Bentchmarks')
plt.show()

