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


