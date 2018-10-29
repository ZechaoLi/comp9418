#Group Member:
#Meng Sun   ID:z5149213
#Zechao Li  ID:z5172016

import zipfile
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import random
import gpflow
from gpflow.test_util import notebook_niter
from sklearn.linear_model import LogisticRegression


def read_train_zip(filename):
    zip_f = zipfile.ZipFile(filename)
    file_name_list = zip_f.namelist()[1:]
    x_file = []
    sub_x = []
    y_i = 0
    data_y = None
    all_x = []

    for file in file_name_list:
        if file[-1] == 'x':
            x_file.append(file)

    for xfile in x_file:
        temp_x_dict = dict()
        file_x_c = zip_f.read(xfile).decode('utf-8').split("\n")[:-1]
        file_y_c = [int(e) for e in zip_f.read(xfile[:-2] + ".y").decode('utf-8').split("\n")[:-1]]

        for e in file_x_c:
            l = e.split(' ')
            if int(l[0]) not in temp_x_dict:
                temp_x_dict[int(l[0])] = [int(l[1])]
            else:
                temp_x_dict[int(l[0])].append(int(l[1]))
        all_x.append(temp_x_dict)
        for e in temp_x_dict:
            sub_x.append(e)

        sub_y_lable = np.zeros(shape=[len(file_y_c), 1])
        for i in range(len(file_y_c)):
            sub_y_lable[i][0] = file_y_c[i]
        if y_i == 0:
            data_y = sub_y_lable
            y_i = 1
        else:
            data_y = np.vstack((data_y, sub_y_lable))

    data_x = lil_matrix((len(sub_x), 2035523))
    k = 0
    for x in all_x:
        for e in x:
            for i in x[e]:
                data_x[k, i - 1] = 1
            k += 1

    return data_x, data_y


def read_test_zip(filename):
    zip_f = zipfile.ZipFile(filename)
    file_name_list = zip_f.namelist()[1:]
    x_file = []
    sub_x = []
    all_x = []
    seperate_list = []
    for file in file_name_list:
        if file[-1] == 'x':
            x_file.append(int(file[20:-2]))

    x_file.sort()
    for xfile in x_file:
        temp_x_dict = dict()
        file_x_c = zip_f.read("conll_test_features/" + str(xfile) + ".x").decode('utf-8').split("\n")[:-1]

        for e in file_x_c:
            l = e.split(' ')
            if int(l[0]) not in temp_x_dict:
                temp_x_dict[int(l[0])] = [int(l[1])]
            else:
                temp_x_dict[int(l[0])].append(int(l[1]))
        seperate_list.append(len(temp_x_dict))
        all_x.append(temp_x_dict)
        for e in temp_x_dict:
            sub_x.append(e)

    data_x = lil_matrix((len(sub_x), 2035523))
    k = 0
    for x in all_x:
        for e in x:
            for i in x[e]:
                data_x[k, i - 1] = 1
            k += 1

    return data_x, seperate_list



def main():
    # read the training data_x, data_y from dataset
    data_x, data_y = read_train_zip("conll_train.zip")
    # transform data_x matrix to sparse csr_matrix
    X_sparse = csr_matrix(data_x)
    # use truncatedSVD to do dimensional reduction
    tsvd = TruncatedSVD(n_components=50, algorithm = 'arpack')
    tsvd.fit(X_sparse)
    X_sparse_tsvd = tsvd.transform(X_sparse)

    lable_dict = dict()
    for i in range(len(data_y)):
        if data_y[i][0] not in lable_dict:
            lable_dict[data_y[i][0]] = [i]
        else:
            lable_dict[data_y[i][0]].append(i)

    random_sample = []

    # extract reasonable number of training data
    for e in lable_dict:
        if len(lable_dict[e]) > 5000:
            k = random.sample(lable_dict[e], 5000)
            random_sample = random_sample + k
        else:
            random_sample = random_sample + lable_dict[e]

    # shuffle training dataset and validation dataset
    test_index = [_ for _ in range(211727)]
    random_test = random.sample(test_index, 10000)
    train_x = X_sparse_tsvd[random_sample]
    train_y = data_y[random_sample]
    val_x = X_sparse_tsvd[random_test]
    val_y = data_y[random_test]

    shuffle_list = np.array([_ for _ in range(train_x.shape[0])])
    np.random.shuffle(shuffle_list)
    train_x = train_x[shuffle_list]
    train_y = train_y[shuffle_list]

    # train the gp model
    g = gpflow.models.SVGP(
        train_x, train_y, kern=gpflow.kernels.RBF(input_dim=50),
        likelihood=gpflow.likelihoods.MultiClass(23),
        minibatch_size=1000,
        Z=train_x[::50].copy(), num_latent=23, whiten=True, q_diag=True)
    opt = gpflow.train.AdamOptimizer()
    opt.minimize(g, maxiter=notebook_niter(2000))

    result_t = g.predict_y(val_x)[0]

    #calculate the ER and MNLP for validation data set with GP model
    c = 0
    for i in range(len(val_x)):
        if result_t[i].argmax() == val_y[i][0]:
            c += 1
    er = 1 - c / len(val_x)

    mnlp = 0
    result_te = np.log(result_t)
    for i in range(len(val_x)):
        for j in range(23):
            mnlp += result_te[i][j]
    mnlp = - mnlp / len(val_x)
    print("GP model:")
    print("error rate: {}, mean negative log probability: {}".format(er, mnlp))

    # calculate the ER and MNLP for validation data set with softmax model
    lgpredict = LogisticRegression(solver='lbfgs', multi_class="multinomial").fit(train_x, train_y)
    lgpresult = lgpredict.predict_proba(val_x)

    c = 0
    for i in range(len(val_x)):
        if lgpresult[i].argmax() == val_y[i][0]:
            c += 1
    er = 1 - c / len(val_x)

    mnlp = 0
    result_te = np.log(lgpresult)
    for i in range(len(val_x)):
        for j in range(22):
            mnlp += result_te[i][j]
    mnlp = - mnlp / len(val_x)
    print("Softmax model:")
    print("error rate: {}, mean negative log probability: {}".format(er, mnlp))


    # read test data from test dataset
    test_x, seperate_list = read_test_zip("conll_test_features.zip")
    # dimensional reduction
    test = tsvd.transform(test_x)

    # predict
    result = g.predict_y(test)[0]
    result_1 = g.predict_y(test[20000:40000])[0]
    index = 0
    result = np.log(result)
    final = ''

    for e in seperate_list:
        for i in range(e):
            for j in range(22):
                final += str(round(result[index + i][j], 8))
                final += ","
            final += str(round(result[i][22], 8))
            final += "\n"
        final += "\n"
        index += e

    # written result prediction.txt
    with open("predictions.txt", "w") as f:
        f.write(final)



if __name__ == '__main__':
    main()
