import sys, re

def main(args):
	for arg in args[1:]:
		f = open(arg + ".h", "r")
		w = open(arg + "_new.h", "w+")
		numeric_const_pattern = '[-+]?[0-9]+\.[0-9]+(?:[eE][-+]?[0-9]+)?'
		rx = re.compile(numeric_const_pattern, re.VERBOSE)
		output = ""
		for line in f.readlines():
			matches = rx.findall(line)
			if len(matches) == 0:
				output += line
				continue

			for match in matches:
				replace = str(int(float(match) * 256))
				if replace[0] == '0' and replace != "0":
					print line
					print matches
				if replace == "06640625":
					print "Hey"
				line = line.replace(match, str(int(round(float(match) * 256))))
			
			output += line

		w.write(output)

if __name__ == "__main__":
	main(sys.argv)