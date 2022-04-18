path = {}

path['172.18.167.6'] = {
    'sr_model': {
        'panasonic': '/data2/xxq/SR/exp/CDC/CDC-X4-panasonic/checkpoint/HGSR-MHR_X4_best.pth',
        'sony': '',
        'P11': ''
    },
    'circle_model': {
        'panasonic': '/opt/ohpc/xxq/SR/result3/experiments_panasonic/circle_cdc_panasonic_no_pretrain_1_1/best_psnr.pth',
        'sony': '/data3/xxq/SR/result/experiments_sony/circle_cdc_sony/best_psnr.pth',
        'P11': '/data3/xxq/SR/result/experiments_olympus/circle_cdc_olympus/best.pth'
    }
}

path['172.18.167.28'] = {
    'sr_model': {
        'panasonic': '/data3/xxq/SR/result/experiments_panasonic/cdc_panasonic/HGSR-MHR_X4_best0.pth',
        'sony': '/media/data0/xxq/SR/exp/result/experiments_sony/cdc_sony_kaiming_init/best_psnr.pth',
        'P11': '/media/data0/xxq/SR/exp/CDC/CDC-X4-olympus/checkpoint/HGSR-MHR_X4_best.pth'
    },
    'circle_model': {
        'panasonic': '/media/data0/xxq/SR/exp/result/experiments_panasonic/circle_cdc_panasonic_no_pretrain_1_1/best_psnr.pth',
        'sony': '/media/data0/xxq/SR/exp/result/experiments_sony/circle_cdc_sony/best_psnr.pth',
        'P11': '/media/data0/xxq/SR/exp/result/experiments_olympus/circle_cdc_olympus/best.pth'
    },
}

path['10.18.26.1'] = {
    'circle_model': {
        'panasonic': '/data2/xxq/SR/exp/experiments/circle_cdc_panasonic_no_pretrain_1_1/best_psnr.pth',
        'sony': '/data2/xxq/SR/exp/experiments/circle_cdc_sony/best_psnr.pth',
        'P11': '/data3/xxq/SR/result/experiments_olympus/circle_cdc_olympus/best.pth'
    }
}