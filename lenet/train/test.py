import os
import pickle
import re
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import workspace

# reset the workspace, to make sure the model is actually loaded
root_folder = "files"
workspace.ResetWorkspace(root_folder)

# verify that all blobs are destroyed.
print("The blobs in the workspace after reset: []".format(workspace.Blobs()))

# load the predict net
predict_net = pe.prepare_prediction_net(os.path.join(root_folder, "mnist_model.minidb"), "minidb")

# verify that blobs are loaded back
print("The blobs in the workspace after loading the model: []".format(workspace.Blobs()))

# feed the previously saved data to the loaded model
data = np.array([[[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01171875, 0.0703125,
		0.0703125, 0.0703125, 0.4921875, 0.53125, 0.68359375, 0.1015625, 0.6484375, 0.99609375, 0.96484375,
		0.49609375, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.1171875, 0.140625, 0.3671875, 0.6015625, 0.6640625,
		0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.87890625, 0.671875, 0.98828125, 0.9453125,
		0.76171875, 0.25, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.19140625, 0.9296875, 0.98828125, 0.98828125, 0.98828125,
		0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98046875, 0.36328125, 0.3203125, 0.3203125,
		0.21875, 0.15234375, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0703125, 0.85546875, 0.98828125, 0.98828125,
		0.98828125, 0.98828125, 0.98828125, 0.7734375, 0.7109375, 0.96484375, 0.94140625, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3125, 0.609375,
		0.41796875, 0.98828125, 0.98828125, 0.80078125, 0.04296875, 0.0, 0.16796875, 0.6015625, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0546875, 0.00390625, 0.6015625, 0.98828125, 0.3515625, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.54296875, 0.98828125, 0.7421875, 0.0078125, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.04296875, 0.7421875, 0.98828125, 0.2734375, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13671875, 0.94140625, 0.87890625,
		0.625, 0.421875, 0.00390625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31640625,
		0.9375, 0.98828125, 0.98828125, 0.46484375, 0.09765625, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.17578125, 0.7265625, 0.98828125, 0.98828125, 0.5859375, 0.10546875, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0625, 0.36328125, 0.984375, 0.98828125, 0.73046875, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.97265625, 0.98828125, 0.97265625,
		0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.1796875, 0.5078125, 0.71484375, 0.98828125, 0.98828125,
		0.80859375, 0.0078125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.15234375, 0.578125, 0.89453125, 0.98828125, 0.98828125, 0.98828125,
		0.9765625, 0.7109375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.09375, 0.4453125, 0.86328125, 0.98828125, 0.98828125, 0.98828125, 0.98828125,
		0.78515625, 0.3046875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.08984375, 0.2578125, 0.83203125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.7734375,
		0.31640625, 0.0078125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0703125, 0.66796875, 0.85546875, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.76171875, 0.3125,
		0.03515625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.21484375,
		0.671875, 0.8828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.953125, 0.51953125, 0.04296875,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0,
		0.53125, 0.98828125, 0.98828125, 0.98828125, 0.828125, 0.52734375, 0.515625, 0.0625, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]]*64, dtype=np.float32) 

def writeHeader(name, *args):
    f = open("../headers/" + name, "w+")
    str_var = ""
    for arg in args:
        arr = workspace.FetchBlob(arg)
        min_arr = np.squeeze(arr)
        str_arr = ",".join( map(str, min_arr.tolist()))
        str_arr = str(str_arr).replace('[', '{').replace(']', '}').replace(',', ', ').replace(',  ', ', ')
        for match in re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str_arr):
        	str_arr = str_arr.replace(match, "F_LIT(" + match + ")")
        str_dim = ""
        for dim in min_arr.shape:
            str_dim += '[' + str(dim) + ']'
        str_var += "signed short " + arg + str_dim + " = {" + str_arr + "};\n\n"

    f.write(str_var)
    f.close()

def compress(name):
	f = open("headers/" + name + ".h", "w+")
	str_var = ""
	arr = workspace.FetchBlob(name + "_w")
	min_arr = np.squeeze(arr)
	flat_arr = min_arr.flatten().reshape(-1, 1)
	min_val = flat_arr.min()
	max_val = flat_arr.max()
	num_buckets = 16
	buckets = np.array([((max_val - min_val) / num_buckets) * x + min_val for x in range(0, num_buckets)]).reshape(-1, 1)
	clusters = KMeans(n_clusters=num_buckets, init=buckets).fit(flat_arr)
	# clusters = KMeans(n_clusters=2).fit(flat_arr)
	labels = clusters.labels_.flatten().reshape(-1, 1600)
	centroids = clusters.cluster_centers_.flatten()
	# Weights
	str_arr = ",".join(map(str, centroids.tolist()))
	str_arr = str(str_arr).replace('[', '{').replace(']', '}').replace(',', ', ').replace(',  ', ', ')
	str_dim = ""
	for dim in centroids.shape:
		str_dim += '[' + str(dim) + ']'

	str_var += "float " + name + "_w" + str_dim + " = {" + str_arr + "};\n\n"

	# Labels
	str_arr = map(str, labels.tolist())
	str_arr = str(str_arr).replace('[', '{').replace(']', '}').replace(',', ', ').replace(',  ', ', ').replace("'", "")
	# print str_arr
	str_dim = ""
	for dim in labels.shape:
		str_dim += '[' + str(dim) + ']'

	str_var += "unsigned char " + name + "_w" + str_dim + " = " + str_arr + ";\n\n"

	# Biases
	arr = workspace.FetchBlob(name + "_b")
	min_arr = np.squeeze(arr)
	str_arr = ",".join(map(str, min_arr.tolist()))
	str_arr = str(str_arr).replace('[', '{').replace(']', '}').replace(',', ', ').replace(',  ', ', ')
	str_dim = ""
	for dim in min_arr.shape:
		str_dim += '[' + str(dim) + ']'

	str_var += "float " + name + "_b" + str_dim + " = {" + str_arr + "};\n\n"

	f.write(str_var)
	f.close()

# compress("fc3")
workspace.FeedBlob("data", data)

f = open('files/mask.summary', 'r')
masked_weights = pickle.load(f)
workspace.FeedBlob("pr_m", masked_weights)
# predict
workspace.RunNetOnce(predict_net)
softmax = workspace.FetchBlob("softmax")
sparse_weights = sparse.csr_matrix(np.multiply(workspace.FetchBlob("pr_w"), masked_weights))

#Dump out the headers
str_var = ""
mats = [(sparse_weights.indices, "pr_idx"), (sparse_weights.indptr, "pr_ptr")]
for arr in mats:
    min_arr = np.squeeze(arr[0])
    str_arr = ",".join(map(str, min_arr.tolist()))
    str_arr = str(str_arr).replace('[', '{').replace(']', '}').replace(',', ', ').replace(',  ', ', ')
    str_dim = ""
    for dim in min_arr.shape:
        str_dim += '[' + str(dim) + ']'
    str_var += "unsigned short " + arr[1] + str_dim + " = {" + str_arr + "};\n\n"
    # print str_var

mats = [(sparse_weights.data, "pr_w")]
for arr in mats:
    min_arr = np.squeeze(arr[0])
    str_arr = ",".join(map(lambda s: "F_LIT(" + s + ")", map(str, min_arr.tolist())))
    str_arr = str(str_arr).replace('[', '{').replace(']', '}').replace(',', ', ').replace(',  ', ', ')
    str_dim = ""
    for dim in min_arr.shape:
        str_dim += '[' + str(dim) + ']'
    str_var += "unsigned short " + arr[1] + str_dim + " = {" + str_arr + "};\n\n"
    # print str_var

f = open("../headers/pr_w.h", "w+")
f.write(str_var)
f.close()

writeHeader("pr.h", "pr_b")
writeHeader("conv1.h", "conv1_w", "conv1_b")
writeHeader("conv2.h", "conv2_w", "conv2_b")
writeHeader("pred.h", "pred_w", "pred_b")

print "Calc Mul: ", np.count_nonzero(sparse_weights.data)
print "PR Mul: ", np.count_nonzero(workspace.FetchBlob('mnist_deploy/Mul'))
print "PR: ", np.count_nonzero(workspace.FetchBlob('pr')[0])
inter = np.multiply(workspace.FetchBlob("pr_w"), masked_weights)
inter = np.dot(inter, workspace.FetchBlob("pool2")[0].flatten())
inter = np.add(inter, workspace.FetchBlob("pr_b"))
inter[inter < 0] = 0.0
print "CALC PR: ", np.count_nonzero(inter)
print workspace.Blobs()

# the first letter should be predicted correctly
output = [str(i) + " : " + str(round(x, 2)) for i, x in enumerate(softmax[0])]
for o in output:
    print o
