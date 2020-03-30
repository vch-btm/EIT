# coding=utf-8

from __future__ import print_function, division
from ML_help import *

from keras.layers import AveragePooling2D
from keras.models import Model

from keras.layers import Input
from keras.callbacks import TensorBoard
import numpy as np

import socket

from shutil import copyfile

#########################################
#########################################

batch_size = 8

# specification of the problem size
img_rows = 64
num_bases = 16

# IPAddr = socket.gethostbyname(socket.gethostname())
comp_name = socket.gethostname()

if "127" in comp_name:
    img_rows = 32
    num_bases = 8

if len(sys.argv) > 2:
    img_rows = int(sys.argv[2])
    if len(sys.argv) > 3:
        num_bases = int(sys.argv[3])

img_cols = img_rows

# num_bases = img_rows * 2

# how many samples for training, testing and validation
num_data_train = 5000
num_data_valid = num_data_train // 10
num_data_test = num_data_train // 10

# further sepcifications
numEpochs = 100000

shuffleData = not True

randomTxt = f"k1_{img_rows}_{num_bases}"

netType = "t"
versionNr = 2

#########################################
#########################################

path = ""
ID = start_ML(randomTxt)


########################################################################################################################


def build_unet(txt_name):
    if True:
        activation = Mish
        act_params = {}

    if not True:
        downConv = convComb
    else:
        downConv = conv
        # downConv = conv_act
        # downConv = conv_both
        # downConv = conv_both_act

    if True:
        upConv = convUp
    else:
        upConv = convT

    _input = Input(shape=img_shape)

    num_rep = int(np.log2(img_rows)) - 1
    num_parts1 = 1
    num_parts2 = num_parts1
    factorDown = 2
    factorMid = 8
    factorUp = 1

    mult = num_bases // 2 ** (num_rep - 2)

    dimension = num_bases // mult
    # mult = img_shape[-1] // dimension

    branch_outputs = []
    d_list = []
    mult_h = mult // 2

    ################# down #################

    x = Crop(3, 0, mult_h)(_input)
    y = Crop(3, -(mult_h + 1), -1)(_input)

    x = downConv(conc([x, y]), factorDown, 3, 2, parts=num_parts1)

    branch_outputs.append(x)

    for i in range(2 * dimension - 1):
        out = Crop(3, i * mult_h, (i + 2) * mult_h)(_input)
        x = downConv(out, factorDown, 3, 2, parts=num_parts1)
        branch_outputs.append(x)

    d_list.append(conc(branch_outputs))

    ################# splitted #################

    for i in range(num_rep - 1):
        new_branches = []

        while len(branch_outputs) > 1:
            x = conc([branch_outputs.pop(), branch_outputs.pop()])
            x = downConv(x, factorDown, 3, 2, parts=num_parts1)
            new_branches.append(x)

        branch_outputs = new_branches
        d_list.append(conc(branch_outputs))  # for skipping

    ################# Mitte #################

    d_last = d_list.pop()
    n_list = [d_last]
    n = d_last

    for _ in range(1):
        n_list.append(conv(n, factorMid, 1, 1, parts=0))
        n = conc(n_list)

    d_list.append(n)
    d_list.append(d_last)

    ################# up #################

    for _ in range(num_rep):
        d = conc([d_list.pop(), d_list.pop()])

        dE = Crop(3, 0, -1, 2)(d)
        dO = Crop(3, -1, 0, -2)(d)

        d_list.append(conc([upConv(dE, factorUp, 3, 2, parts=num_parts2), upConv(dO, factorUp, 3, 2, parts=num_parts2)]))

    for i in range(0):
        d1 = d_list.pop()
        d2 = conv(d1, factorUp, 3, 1, parts=num_parts2)
        d_list.append(conc([d1, d2]))

    _output = conv(conc(d_list), 1, 1, 1, parts=num_parts2)

    return Model(_input, _output, name=txt_name)


########################################################################################################################


img_shape = (img_rows, img_cols, num_bases)

# optimizer = RAdam1()
optimizer = Lookahead(RAdam2(warmup_proportion=0.1, min_lr=1e-5))

model = build_unet("unet")
model.summary()

model.compile(loss='mse', optimizer=optimizer)
# lookahead = Lookahead(k=3, alpha=0.5)  # Initialize Lookahead
# lookahead.inject(model)  # add into model

ID = f"{ID}_{model.count_params()}"

#########################

pathData = f"{path}MLdata/{img_rows}x{img_cols}/k1_{img_rows}"

# load data
numDataZ = 6000

# all_sets = load_data_bases(pathData, numDataZ, num_data_train, num_data_valid, num_data_test, num_bases, shuffleData)
all_sets = load_data_bases_borders(pathData, numDataZ, num_data_train, num_data_valid, num_data_test, num_bases, shuffleData)

img_shape = all_sets[0].shape[1:]

#########################


subpathID = f"{randomTxt}/{ID}"
os.makedirs(f'{subpathID}', exist_ok=True)
file_name = os.path.basename(sys.argv[0])
# plot_model(net, to_file=f'{subpathID}/{ID}unet.png', show_shapes=True)
copyfile(file_name, f"{subpathID}/{file_name}")
copyfile("ML_help.py", f"{subpathID}/ML_help.py")

tensorboard = TensorBoard(log_dir=f"{path}logs/{randomTxt}/{ID}_{netType}")
tensorboard.set_model(model)

images_to_sample = [int(num_data_test * 0.25), int(num_data_test * 0.5), int(num_data_test * 0.75), int(num_data_test * 0.99)]

train(5000000000, model, num_data_train, all_sets, img_shape, img_rows, img_cols, tensorboard, images_to_sample, path, ID, subpathID, netType, versionNr, batch_size=batch_size, sample_interval=100, start_epoch=0, change_after=15, max_bs=256)


def build_unet_old(txt_name):
    if True:
        activation = Mish
        act_params = {}

    if not True:
        downConv = convComb
    else:
        downConv = conv

    if not True:
        upConv = convUp
    else:
        upConv = convT

    d0 = Input(shape=img_shape)

    d = BatchNormalization(d0)

    use_overlap = True

    num_rep = int(np.log2(img_rows)) - 2
    num_parts1 = 0
    num_parts2 = 0
    num_dep_rep = 2
    factor = 2

    max_dil = 2

    mult = 4
    dimension = img_shape[-1] // mult
    # mult = img_shape[-1] // dimension

    branch_outputs = []

    start_list = []

    if use_overlap:
        mult_h = mult // 2

        x = conv(Crop(3, 0, mult_h)(d), 1 * factor, 3, 1, parts=num_parts1)

        branch_outputs.append(x)
        start_list.append(x)

        for i in range(2 * dimension - 1):
            out = Crop(3, i * mult_h, (i + 2) * mult_h)(d)

            x = conv(out, 1 * factor, 3, 1, parts=num_parts1)
            branch_outputs.append(x)
            start_list.append(x)

        # branch_outputs.append(conv(Crop(3, -(mult_h + 1), -1)(d0), 1 * factor, 3, 1, parts=num_parts1))

        # branch_outputs.reverse()

        while len(branch_outputs) > 0:
            new_branches = []

            if len(branch_outputs) == 2:
                x = conv(conc([branch_outputs.pop(), branch_outputs.pop()]), 1 * factor, 3, 1, parts=num_parts1)
                branch_outputs.append(x)
                start_list.append(x)
                break

            while len(branch_outputs) > 1:
                x = conv(conc([branch_outputs.pop(), branch_outputs.pop()]), 1 * factor, 3, 1, parts=num_parts1)
                new_branches.append(x)
                # start_list.append(x)

            if len(branch_outputs) == 1:
                new_branches.append(branch_outputs.pop())

            branch_outputs = new_branches

        temp = branch_outputs.pop()
    else:
        for i in range(dimension):
            out = Crop(3, i * mult, (i + 1) * mult)(d0)
            branch_outputs.append(conv(out, 1 * factor, 3, 1, parts=num_parts1))

        temp = conc(branch_outputs)

    d_list = [temp]

    for _ in range(num_rep):
        for _ in range(num_dep_rep):
            # d_list.append(downConv(d_list[-1], 4 * factor, 3, 1, parts=num_parts1, dil_max=max_dil))
            d_list.append(conv(d_list.pop(), 4 * factor, 3, 1, parts=num_parts1, dil_max=1))

        d_list.append(downConv(d_list[-1], 2 * factor, 3, 2, parts=num_parts1, dil_max=max_dil))

    d_last = d_list.pop()
    n_list = [d_last]
    n = d_last

    for _ in range(1):
        n_list.append(conv(n, 8 * factor, 1, 1, parts=0))
        n = conc(n_list)

    d_list.append(n)
    d_list.append(d_last)

    for _ in range(num_rep):
        d = conc([d_list.pop(), d_list.pop()])
        d_list.append(upConv(d, 4 * factor, 3, 2, parts=num_parts2))
        for _ in range(num_dep_rep):
            # d = conc([d_list.pop(), d_list.pop()])
            d = d_list.pop()
            # d_list.append(upConv(d, 4 * factor2, 3, 1, parts=num_parts2))
            d_list.append(conv(d, 4 * factor, 3, 1, parts=num_parts2))

    for _ in range(0):
        d1 = d_list.pop()
        d2 = conv(d1, 1 * factor, 3, 1, parts=num_parts2)
        d_list.append(conc([d1, d2]))

    [d_list.append(i) for i in start_list]

    # fin = conv(d_list.pop(), 1, 1, 1, parts=num_parts2)
    fin = conv(conc(d_list), 1, 1, 1, parts=num_parts2)

    return Model(d0, fin, name=txt_name)
