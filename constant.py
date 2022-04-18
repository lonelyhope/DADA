import socket

def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

IP = get_host_ip()
print('IP:', IP)

# Merge IMG and Canon
CAMERA = ['DSC', 'panasonic', 'sony', 'Canon', 'IMG', 'P11']
SEED = 1    # random seed
VAL_NUM = 50   # 对于每个 camera，train set 中前 VAL_NUM 个 pair 默认不用，留作 valid set 使用
 
# DATA_ROOT = "/opt/ohpc/xxq/SR"

if (IP == '172.18.167.6'):
    TRN_PATH = "/opt/ohpc/xxq/SR/DRealSR/x4/Train_x4"
    TST_PATH = "/opt/ohpc/xxq/SR/DRealSR/x4/Test_x4"
    SAVE_ROOT = "/opt/ohpc/xxq/SR/result3"
    SAVE_ROOT = "/data3/xxq/SR/result"
    SAVE_ROOT = "/opt/ohpc/xxq/SR/result_1_10"
    SAVE_ROOT = "/data/xxq/SR/result_generalization"
elif (IP == '172.18.167.28'):
    # TRN_PATH = "/media/data1/RealSR/Dataset/Photo-4/train_patch"
    # TST_PATH = "/media/data1/RealSR/Dataset/Photo-4/Realvalid93"
    TRN_PATH = "/media/data2/RealSR/Dataset/Photo-4/train_patch"
    TST_PATH = "/media/data2/RealSR/Dataset/Photo-4/Realvalid93"
    SAVE_ROOT = "/media/data4/xxq/SR/exp/result"
    SAVE_ROOT = "/media/data4/xxq/SR/exp/result2"
    # SAVE_ROOT = "/media/data4/xxq/SR/exp/result_generalization"
    # SAVE_ROOT = "/media/data4/xxq/SR/exp/result_review"
    # SAVE_ROOT = "/media/data4/xxq/SR/exp/result_generalization/experiments_gip"
    SAVE_ROOT = "/media/data0/xxq/SR/exp/res"
elif (IP == '10.18.26.1'):
    TRN_PATH = '/data2/xxq/SR/dataset/DRealSR/x4/Train_x4'
    TST_PATH = '/data2/xxq/SR/dataset/DRealSR/x4/Test_x4'
    SAVE_ROOT = '/data2/xxq/SR/exp'

