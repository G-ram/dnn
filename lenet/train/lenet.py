import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe
from IPython import display
from tqdm import tqdm

from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.python.modeling.initializers import Initializer
from caffe2.python import core, model_helper, workspace, brew
from scipy import stats
import pickle

# This section preps your image and test set in a lmdb database
def DownloadResource(url, path):
    '''Downloads resources from s3 by url and unzips them to the provided path'''
    import requests, zipfile, StringIO
    print("Downloading... {} to {}".format(url, path))
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall(path)
    print("Completed download and extraction.")

def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label

def custom_FC(
    model, op_call, blob_in, blob_out, dim_in, dim_out, weight_init=None,
        bias_init=None, WeightInitializer=None, BiasInitializer=None,
        enable_tensor_core=False, **kwargs
):
    WeightInitializer = initializers.update_initializer(
        WeightInitializer, weight_init, ("XavierFill", {})
    )
    BiasInitializer = initializers.update_initializer(
        BiasInitializer, bias_init, ("ConstantFill", {})
    )
    if not model.init_params:
        WeightInitializer = initializers.ExternalInitializer()
        BiasInitializer = initializers.ExternalInitializer()

    blob_out = blob_out or model.net.NextName()
    bias_tags = [ParameterTags.BIAS]
    if 'freeze_bias' in kwargs:
        bias_tags.append(ParameterTags.COMPUTED_PARAM)

    weight = model.create_param(
        param_name=blob_out + '_w',
        shape=[dim_out, dim_in],
        initializer=WeightInitializer,
        tags=ParameterTags.WEIGHT
    )

    mask_init = np.ones((500, 1600), dtype=np.float32)
    mask = model.create_param(
        param_name=blob_out + '_m',
        shape=[dim_out, dim_in],
        initializer=Initializer(operator_name='GivenTensorFill', values=mask_init),
        tags=ParameterTags.COMPUTED_PARAM
    )

    bias = model.create_param(
        param_name=blob_out + '_b',
        shape=[dim_out, ],
        initializer=BiasInitializer,
        tags=bias_tags
    )

    # enable TensorCore by setting appropriate engine
    if enable_tensor_core:
        kwargs['engine'] = 'TENSORCORE'

    blob_inter = model.net.Mul([mask,  weight])
    return op_call([blob_in, blob_inter, bias], blob_out, **kwargs)

def fc_sp(model, blob_in, blob_out, *args, **kwargs):
    return custom_FC(model, model.net.FC, blob_in, blob_out, *args, **kwargs)

def AddLeNetModel(model, data):
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=100, kernel=5)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    # fc3 = brew.fc(model, pool2, 'fc3', dim_in=100 * 4 * 4, dim_out=500)

    pr = brew.fc_sp(model, pool2, 'pr', dim_in=100 * 4 * 4, dim_out=500)

    relu = brew.relu(model, pr, pr)
    pred = brew.fc(model, relu, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax

def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy

def AddTrainingOperators(model, softmax, label):
    """Adds training operators to the model."""
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = brew.iter(model, "iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - ModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)

def AddBookkeepingOperators(model):
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)

def main():
    brew.Register(fc_sp)
    # If you would like to see some really detailed initializations,
    # you can change --caffe2_log_level=0 to --caffe2_log_level=-1
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    print("Necessities imported!")

    # current_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks')
    data_folder = 'data'
    root_folder = 'files'
    db_missing = False

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)   
        print("Your data folder was not found!! This was generated: {}".format(data_folder))

    # Look for existing database: lmdb
    if os.path.exists(os.path.join(data_folder,"mnist-train-nchw-lmdb")):
        print("lmdb train db found!")
    else:
        db_missing = True

    if os.path.exists(os.path.join(data_folder,"mnist-test-nchw-lmdb")):
        print("lmdb test db found!")
    else:
        db_missing = True

    # attempt the download of the db if either was missing
    if db_missing:
        print("one or both of the MNIST lmbd dbs not found!!")
        db_url = "http://download.caffe2.ai/databases/mnist-lmdb.zip"
        try:
            DownloadResource(db_url, data_folder)
        except Exception as ex:
            print("Failed to download dataset. Please download it manually from {}".format(db_url))
            print("Unzip it and place the two database folders here: {}".format(data_folder))
            raise ex

    if os.path.exists(root_folder):
        print("Looks like you ran this before, so we need to cleanup those old files...")
        shutil.rmtree(root_folder)

    os.makedirs(root_folder)
    workspace.ResetWorkspace(root_folder)

    print("training data folder:" + data_folder)
    print("workspace root folder:" + root_folder)

    arg_scope = {"order": "NCHW"}
    train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)
    data, label = AddInput(
        train_model, batch_size=64,
        db=os.path.join(data_folder, 'mnist-train-nchw-lmdb'),
        db_type='lmdb')
    softmax = AddLeNetModel(train_model, data)
    AddTrainingOperators(train_model, softmax, label)
    AddBookkeepingOperators(train_model)

    test_model = model_helper.ModelHelper(
        name="mnist_test", arg_scope=arg_scope, init_params=False)
    data, label = AddInput(
        test_model, batch_size=100,
        db=os.path.join(data_folder, 'mnist-test-nchw-lmdb'),
        db_type='lmdb')
    softmax = AddLeNetModel(test_model, data)
    AddAccuracy(test_model, softmax, label)

    # Deployment model. We simply need the main LeNetModel part.
    deploy_model = model_helper.ModelHelper(
        name="mnist_deploy", arg_scope=arg_scope, init_params=False)
    AddLeNetModel(deploy_model, "data")

    with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:
        fid.write(str(train_model.net.Proto()))
    with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
        fid.write(str(train_model.param_init_net.Proto()))
    with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:
        fid.write(str(test_model.net.Proto()))
    with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:
        fid.write(str(test_model.param_init_net.Proto()))
    with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
        fid.write(str(deploy_model.net.Proto()))
    print("Protocol buffers files have been created in your root folder: " + root_folder)

    # The parameter initialization network only needs to be run once.
    workspace.RunNetOnce(train_model.param_init_net)
    # creating the network
    workspace.CreateNet(train_model.net, overwrite=True)
    # set the number of iterations and track the accuracy & loss
    init_train_iters = 200
    secondary_train_iters = 100
    test_iters = 100
    accuracy = np.zeros(init_train_iters + secondary_train_iters)
    loss = np.zeros(init_train_iters + secondary_train_iters)
    # Now, we will manually run the network for 200 iterations.
    for i in tqdm(range(init_train_iters)):
        workspace.RunNet(train_model.net)
        accuracy[i] = workspace.FetchBlob('accuracy')
        loss[i] = workspace.FetchBlob('loss')

    # Mask
    threshold = 0.045
    weights = workspace.FetchBlob("pr_w")
    masked_weights = stats.threshold(weights, threshmin=-threshold, threshmax=threshold, newval=1.0)
    masked_weights[masked_weights != 1.0] = 0.0
    f = open('files/mask.summary', 'w+')
    pickle.dump(masked_weights, f)
    f.close()
    nonzero = float(np.count_nonzero(masked_weights))
    total = float(masked_weights.size)
    print "Nonzero: ", nonzero, " Total:", total, " NonZero / Total: ", nonzero / total

    # Secondary Training
    workspace.FeedBlob("pr_m", masked_weights)
    for i in tqdm(range(secondary_train_iters)):
        workspace.RunNet(train_model.net)
        accuracy[i + init_train_iters] = workspace.FetchBlob('accuracy')
        loss[i + init_train_iters] = workspace.FetchBlob('loss')

    # run a test pass on the test net
    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)
    test_accuracy = np.zeros(test_iters)

    workspace.FeedBlob("pr_m", masked_weights)
    for i in tqdm(range(test_iters)):
        workspace.RunNet(test_model.net.Proto().name)
        test_accuracy[i] = workspace.FetchBlob('accuracy')
    # After the execution is done, let's plot the values.
    print('test_accuracy: %f' % test_accuracy.mean())

    pe_meta = pe.PredictorExportMeta(
        predict_net=deploy_model.net.Proto(),
        parameters=[str(b) for b in deploy_model.params],
        inputs=["data", "pr_m"],
        outputs=["softmax"],
    )

    # save the model to a file. Use minidb as the file format
    pe.save_to_db("minidb", os.path.join(root_folder, "mnist_model.minidb"), pe_meta)
    print("The deploy model is saved to: " + root_folder + "/mnist_model.minidb")

if __name__ == "__main__":
    main()