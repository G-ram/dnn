width = 28
height = 28
nth = 4
def main():
	f = open("train-images-idx3-ubyte", 'rb')
	raw = f.read().encode('hex')
	chunks = [raw[i : i + 2] for i in range(0, len(raw), 2)]
	pixels = chunks[16:]
	img = []
	for i in range(width * height * nth, width * height * (nth + 1)):
		img.append(pixels[i])

	output = "{{"
	carriage = 0
	new_arr = 1
	for p in img:
		if new_arr == 28:
			output += str(float(int(p, 16)) / 256.) + "}, {"
			new_arr = 0
		else:
			output += str(float(int(p, 16)) / 256.) + ", "

		if carriage == 9:
			output += "\n"
			carriage = 0

		new_arr += 1
		carriage += 1

	output = "float input[28][28] = " + output[:-5] + "}};"
	print output

	f = open("train-labels-idx1-ubyte", 'rb')
	raw = f.read().encode('hex')
	chunks = [raw[i : i + 2] for i in range(0, len(raw), 2)]
	labels = chunks[8:]
	print "Label: ", int(labels[nth], 16)

if __name__ == "__main__":
	main()