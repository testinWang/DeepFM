# -*- coding:utf-8 -*-
"""
-对数值型特征，normalize处理
-对类别型特征，对长尾(出现频次低于200)的进行过滤
"""
import os
import sys
import random
import collections

# 13个连续型列，26个类别型列
continous_features = range(1, 14)
categorial_features = range(14, 40)

# 对连续值进行截断处理(取每个连续值列的95%分位数)
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]

class CategoryDictGenerator:
    """
    类别型特征编码字典
    """
    def __init__(self, num_feature):
        self.dicts = [] #类别列表
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int)) #num_feature 个元素的list ,元素类型为字典，value为int类型

    def build(self, datafile, categorial_features, cutoff=0):
        #*************************************
        '''
        dicts 的 格式 dicts 本身为一个长度为26的list 其中每个元素为字典，dict[1]为原始连续特征一列
        对应的所有的one-hot为键，数量为value的字典
        此步骤的作用： 统计每一列中的非空值数
        '''
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')#rstrip 删除最右边的指定字符
                #print(len(features))
                for i in range(0, self.num_feature):      #nus_feature =26 个类别特征
                    if features[categorial_features[i]] != '': #categorial_features = range(14, 40)
                        self.dicts[i][features[categorial_features[i]]] += 1
                        if i ==1:
                            print(i,self.dicts)
        #*******************************************
        for i in range(0, self.num_feature):
            #filter中的函数参数是以每个元素为输入
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items()) #过滤掉，特征出现次数少于cutoff的特征
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return list(map(len, self.dicts)) #返回每一个字典的大小

class ContinuousFeatureGenerator:
    """
    对连续值特征做最大最小值normalization
    """
    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature

    def build(self, datafile, continous_features):#搜索每一种特征的最大值和最小值
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > continous_clip[i]:
                            val = continous_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0 #值为空，则用0填充
        val = float(val)
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx]) #归一化到0-1

def preprocess(input_dir, output_dir):
    """
    对连续型和类别型特征进行处理
    """
    dists = ContinuousFeatureGenerator(len(continous_features))#continous_features = range(1, 14) 13
    dists.build(input_dir + 'train.txt', continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))  #26
    dicts.build(input_dir + 'train.txt', categorial_features, cutoff=150)

    output = open(output_dir + 'feature_map', 'w')
    for i in continous_features:    #continous_features = range(1, 14)
        output.write("{0} {1}\n".format('I' + str(i), i))
    dict_sizes = dicts.dicts_sizes()
    categorial_feature_offset = [dists.num_feature]
    for i in range(1, len(categorial_features) + 1):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)
        for key, val in dicts.dicts[i - 1].items():
            output.write("{0} {1}\n".format('C' + str(i) + '|' + key, categorial_feature_offset[i - 1] + val + 1))

    random.seed(0)

    # 90%的数据用于训练，10%的数据用于验证
    with open(output_dir + 'tr.libsvm', 'w') as out_train:
        with open(output_dir + 'va.libsvm', 'w') as out_valid:
            with open(input_dir + 'train.txt', 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')

                    feat_vals = []
                    for i in range(0, len(continous_features)): #14
                        val = dists.gen(i, features[continous_features[i]])
                        feat_vals.append(
                            str(continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                    for i in range(0, len(categorial_features)):
                        val = dicts.gen(i, features[categorial_features[i]]) + categorial_feature_offset[i]
                        feat_vals.append(str(val) + ':1')

                    label = features[0]
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write("{0} {1}\n".format(label, ' '.join(feat_vals)))
                    else:
                        out_valid.write("{0} {1}\n".format(label, ' '.join(feat_vals)))

    with open(output_dir + 'te.libsvm', 'w') as out:
        with open(input_dir + 'test.txt', 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                feat_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    feat_vals.append(str(continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i] - 1]) + categorial_feature_offset[i]
                    feat_vals.append(str(val) + ':1')

                out.write("{0} {1}\n".format(label, ' '.join(feat_vals)))
if __name__ =="__main__":
    input_dir = 'F:\CTR_data\criteo_data\\' #读入的数据已经经过one-hot处理，每行为40个数据
    if not os.path.exists('./criteo_data/'):
        os.makedirs('./criteo_data/')
    output_dir = './criteo_data/'
    print("将数据加工成libsvm格式.....")
    preprocess(input_dir, output_dir)