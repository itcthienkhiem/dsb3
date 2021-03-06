import numpy as np
import hashlib

import utils
import utils_lung
import pathfinder
#基本思路

    # 1. 把train, val, test的病人id统计到all_pids中；
    # 2. 把标注的csv中的病人统计到n_patients中；
    # 3. 把n_patients中正样本的15%和负样本的15%作为整体的测试样本final_test；
    # 4. 把all_pids中排除掉n_patients部分，统计出final_pos_train和final_neg_train，整体为final_train；
    # 5. 把final_train, final_test存入到"final_split.pkl"中；
    
#计算机实现的随机数生成通常为伪随机数生成器，为了使得具备随机性的代码最终的结果可复现，需要设置相同的种子值；
#使用 np.random.RandomState()获取随机数生成器
rng = np.random.RandomState(42)


tvt_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)#utils.load_pkl：返回一个pickle对象
train_pids, valid_pids, test_pids = tvt_ids['training'], tvt_ids['validation'], tvt_ids['test']
all_pids = train_pids + valid_pids + test_pids
print 'total number of pids', len(all_pids)#pid在这里代表的到底是什么？PID是病人ID；要搞明白这里是怎么放置的，以便存放自己的数据

id2label = utils_lung.read_labels(pathfinder.LABELS_PATH)#id2label是一个字典，键是id，值是label
id2label_test = utils_lung.read_test_labels(pathfinder.TEST_LABELS_PATH)
id2label.update(id2label_test)#字典的update用法，把字典id2label_test添加到id2label字典中
n_patients = len(id2label)

pos_ids = []#存储癌症患者的病人ID
neg_ids = []#存储非癌症患者的病人ID

for pid, label in id2label.iteritems():
    if label ==1 :
        pos_ids.append(pid)
    elif label == 0 :
        neg_ids.append(pid)
    else:
        raise ValueError("weird shit is going down")

pos_ratio = 1. * len(pos_ids) / n_patients #计算癌症患者在病人当中所占的比例
print 'pos id ratio', pos_ratio

split_ratio = 0.15
n_target_split = int(np.round(split_ratio*n_patients))#返回浮点数的四舍五入值，计算分割出来的病人数目
print 'given split ratio', split_ratio
print 'target split ratio', 1. * n_target_split / n_patients

n_pos_ftest = int(np.round(split_ratio*len(pos_ids)))#从癌症病人中分割
n_neg_ftest = int(np.round(split_ratio*len(neg_ids)))#从非癌症病人中分割

final_pos_test = rng.choice(pos_ids,n_pos_ftest, replace=False)
final_neg_test = rng.choice(neg_ids,n_neg_ftest, replace=False)
final_test = np.append(final_pos_test,final_neg_test)
print 'pos id ratio final test set', 1.*len(final_pos_test) / (len(final_test)) 

final_train = []
final_pos_train = []
final_neg_train = []
for pid in all_pids:
    if pid not in final_test:
        final_train.append(pid)
        if id2label[pid] == 1:
            final_pos_train.append(pid)
        elif id2label[pid] == 0:
            final_neg_train.append(pid)
        else:
            raise ValueError("weird shit is going down")



print 'pos id ratio final train set', 1.*len(final_pos_train) / (len(final_train))  
print 'final test/(train+test):', 1.*len(final_test) / (len(final_train) + len(final_test))

concat_str = ''.join(final_test)
print 'md5 of concatenated pids:', hashlib.md5(concat_str).hexdigest()

output = {'train':final_train, 'test':final_test}
output_name = pathfinder.METADATA_PATH+'final_split.pkl'
utils.save_pkl(output, output_name)
print 'final split saved at ', output_name





