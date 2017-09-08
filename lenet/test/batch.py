width = 28
height = 28
offset = 4 # imgs to start into the file
num = 20 # number of imgs to take
def main():
	f = open("test-images", 'rb')
	raw = f.read().encode('hex')
	chunks = [raw[i : i + 2] for i in range(0, len(raw), 2)]
	pixels = chunks[16:]
	data = []
	for i in range(width * height * offset, width * height * (offset + 1) * num):
		data.append(pixels[i])
	
	count = 0
	output = ""
	for i in range(0, num):
		carriage = 0
		new_arr = 1
		output += "{{"
		for p in data[width * height * i: width * height * (i + 1)]:
			if new_arr == 28:
				output += str(float(int(p, 16)) * 1. / 256.) + "}, {"
				new_arr = 0
			else:
				output += str(float(int(p, 16)) * 1. / 256.) + ", "

			if carriage == 9:
				output += "\n"
				carriage = 0

			new_arr += 1
			carriage += 1
			count += 1

		output = output[:-4] + "}, \n"
		print count

	out_input = "float input[" + str(num) + "][28][28] = {" + output[:-3] + "};"

	f = open("test-labels", 'rb')
	raw = f.read().encode('hex')
	chunks = [raw[i : i + 2] for i in range(0, len(raw), 2)]
	labels = chunks[8:]
	labels = map(lambda x: int(x, 16), labels[offset: offset + num])
	out_labels = "{"
	for i, label in enumerate(labels):
		out_labels += str(label) + ", "
		if (i + 1) % 10 == 0:
			out_labels += "\n"

	out_labels = "unsigned int labels[" + str(num) + "] = " + out_labels[:-3] + "};\n\n" 
	# print out_labels

	out_num = "unsigned int num = " + str(num) + ";\n\n"

	f = open("../headers/input.h", "w+")
	out = out_num + out_labels + out_input
	f.write(out)
	f.close()

if __name__ == "__main__":
	main()