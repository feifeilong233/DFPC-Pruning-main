import numpy as np


# 0706更新，使用10通道数据，并且加入数据对称增强
def data_combine(tran_to_tags=False, tran_to_highspe=False, tran_to_highdim=False, use_reverse=False,
                 tran_to_value=False, data_clean=False,tran_to_classification=False):

    input1 = np.load("Data_UCTwithDemo/input1.npy")
    output1 = np.load("Data_UCTwithDemo/output1.npy")
    record1 = np.load("Data_UCTwithDemo/record1.npy")
    result1 = np.load("Data_UCTwithDemo/result1.npy")
    for i in range(2,47):
        input2 = np.load("Data_UCTwithDemo/input"+str(i)+'.npy')
        output2 = np.load("Data_UCTwithDemo/output" + str(i) + '.npy')
        record2 = np.load("Data_UCTwithDemo/record" + str(i) + '.npy')
        result2 = np.load("Data_UCTwithDemo/result" + str(i) + '.npy')

        input1 = np.concatenate((input1,input2),0)
        output1 = np.concatenate((output1, output2), 0)
        record1 = np.concatenate((record1, record2), 0)
        result1 = np.concatenate((result1, result2), 0)
    for i in range(1, 47):
        input2 = np.load("Data_UCTwithDemo_same21HZ/input" + str(i) + '.npy')
        output2 = np.load("Data_UCTwithDemo_same21HZ/output" + str(i) + '.npy')
        record2 = np.load("Data_UCTwithDemo_same21HZ/record" + str(i) + '.npy')
        result2 = np.load("Data_UCTwithDemo_same21HZ/result" + str(i) + '.npy')

        input1 = np.concatenate((input1, input2), 0)
        output1 = np.concatenate((output1, output2), 0)
        record1 = np.concatenate((record1, record2), 0)
        result1 = np.concatenate((result1, result2), 0)
    for i in range(1,64):
        input2 = np.load("Data_UCTwithDemo_same/input"+str(i)+'.npy')
        output2 = np.load("Data_UCTwithDemo_same/output" + str(i) + '.npy')
        record2 = np.load("Data_UCTwithDemo_same/record" + str(i) + '.npy')
        result2 = np.load("Data_UCTwithDemo_same/result" + str(i) + '.npy')

        input1 = np.concatenate((input1,input2),0)
        output1 = np.concatenate((output1, output2), 0)
        record1 = np.concatenate((record1, record2), 0)
        result1 = np.concatenate((result1, result2), 0)
    Data = input1
    Label = output1
    record = record1
    result = result1


    # input1 = np.load("Data_UCTwithDemo/input1.npy")
    # output1 = np.load("Data_UCTwithDemo/output1.npy")
    # record1 = np.load("Data_UCTwithDemo/record1.npy")
    # result1 = np.load("Data_UCTwithDemo/result1.npy")
    # result1 = result1[0:1]
    # for i in range(31,47):
    #     input2 = np.load("Data_UCTwithDemo/input"+str(i)+'.npy')
    #     output2 = np.load("Data_UCTwithDemo/output" + str(i) + '.npy')
    #     record2 = np.load("Data_UCTwithDemo/record" + str(i) + '.npy')
    #     result2 = np.load("Data_UCTwithDemo/result" + str(i) + '.npy')
    #     result2 = result2[0:10]
    #
    #     input1 = np.concatenate((input1,input2),0)
    #     output1 = np.concatenate((output1, output2), 0)
    #     record1 = np.concatenate((record1, record2), 0)
    #     result1 = np.concatenate((result1, result2), 0)
    #
    # Data = input1
    # Label = output1
    # record = record1
    # result = result1



    # RecordMyFault = 0
    # for index, data in enumerate(Data):
    #     matrix = data[0]
    #     FinalList = matrix[:, 4]
    #     for a in FinalList:
    #         if a > 0:
    #             RecordMyFault +=1
    #             LabelMatrix = Label[index]
    #             Target = LabelMatrix[a-1,:]
    #             Label[index][a-1, 1] = Target[0]
    #             Label[index][a-1, 0] = float(0)


    # if data_clean is True:
    #     for index, myoutput in enumerate(Label):

    Label = Label.tolist()
    for matrixs in Label:
        for list in matrixs:
            list.append(3 - sum(list))
    Label = np.array(Label)
    if use_reverse is True:
        record = np.concatenate((record, record), 0)
        result = np.concatenate((result, result), 0)
        newLabel = []
        newLabel = Label.tolist()
        for matrix in newLabel:
            for line in matrix:
                tempt = line[1]
                line[1] = line[0]
                line[0] = tempt
        newLabel = np.array(newLabel)
        Label = np.concatenate((Label, newLabel), 0)
        Datanew = Data.tolist()
        for channels in Datanew:
            for matrix in channels:
                tempt = np.array(matrix)
                tempt = tempt.transpose()
                tempt = tempt.tolist()
                matrix = tempt
        Datanew = np.array(Datanew)
        Data = np.concatenate((Data, Datanew), 0)

    if tran_to_highdim is not False:  # 注意是向后插入
        newData = []
        a = tran_to_highdim
        judge_flag = 0
        position = 0
        for i, matrixs in enumerate(Data):

            # while((position-(a-1))<0)
            #     add_matrix1 = np.zeros([(a-position-1)*2, 5, 5])
            #     add_matrix2 =
            #     matrixs = np.concatenate((matrixs, add_matrix), 0)
            #     position += 1
            if position == 0:
                add_matrix = np.zeros([(a - 1) * 2, 5, 5])
                matrixs = np.concatenate((matrixs, add_matrix), 0)
                position += 1
            elif position == 1:
                add_matrix = np.concatenate((Data[i - 1], np.zeros([(a - 2) * 2, 5, 5])), 0)
                matrixs = np.concatenate((matrixs, add_matrix), 0)
                position += 1
            elif position == 2:
                matrix = np.concatenate((Data[i - 1], Data[i - 2]), 0)
                add_matrix = np.concatenate((matrix, np.zeros([(a - 3) * 2, 5, 5])), 0)
                matrixs = np.concatenate((matrixs, add_matrix), 0)
                position += 1
            elif position == 3:
                add_matrix = np.concatenate((Data[i - 1], Data[i - 2]), 0)
                add_matrix = np.concatenate((add_matrix, Data[i - 3]), 0)
                add_matrix = np.concatenate((add_matrix, np.zeros([(a - 4) * 2, 5, 5])), 0)
                matrixs = np.concatenate((matrixs, add_matrix), 0)
                position += 1
            elif position >= 4:
                add_matrix = np.concatenate((Data[i - 1], Data[i - 2]), 0)
                add_matrix = np.concatenate((add_matrix, Data[i - 3]), 0)
                add_matrix = np.concatenate((add_matrix, Data[i - 4]), 0)
                matrixs = np.concatenate((matrixs, add_matrix), 0)
                position += 1
            matrixs = matrixs.tolist()
            newData.append(matrixs)
            if record[i] != judge_flag:
                position = 0
                judge_flag += 1
        Data = np.array(newData)
    if tran_to_tags is True:
        for i, matrixs in enumerate(Label):
            for j, list in enumerate(matrixs):
                sort1 = np.argsort(list)
                list = sort1
                Label[i][j] = list

    if tran_to_highspe is True:  # 试用多种加大特征函数
        Label *= 100

    if tran_to_value is True:
        for indx, i in enumerate(result):
            if i == 1:
                result[indx] = 1
            if i == 2:
                result[indx] = -1

    Label = Label.reshape(Label.shape[0], -1)
    Label = Label.tolist()
    for indx, records in enumerate(record):
        Label[indx].append(result[records])
    Label = np.array(Label)
    return Data, Label, record, result
Data, Label, record, result=data_combine(False,False,5,True,True,True)
m=np.size(Label,0)
n=np.size(Label,1)


