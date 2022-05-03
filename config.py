""" config.py """
from easydict import EasyDict as ed


stitching_config = ed({
    # 图像格式备选
    "image_format_list": ['jpg', 'png', 'bmp', 'tif', 'gif', 'jpeg', ''],
    # 结果的保存格式
    "imwrite_format": 'jpg',

    # 图像压缩大小(一个通道内至多有多少像素)
    "image_size": 400000,
    # 图像裁剪方式
    # "cut_style_list": ['all', 'reel'],
    "cut_style": 'all',
    # 图像拼接方式：
    # "stitch_style": ['successive', 'group'],
    "stitch_style": 'successive',
    # 依次拼接顺序
    # "stitch_order": ['Sequence', 'middle_to_both_sides'],
    "stitch_order": 'Sequence',

    # 是否进行柱面变换
    "cylindrical_projection_run": False,
    # 柱面变换是否进行双线性插值
    "bilinear_interpolation": False,

    # 是否进行多波段融合
    "multi_band_blend": False,
    # 多波段融合图像金字塔层数
    # "pyr_levels": 'max',
    "pyr_levels": 3,

    # 是否画出image_matched
    "image_matched_run": True,
    # image_matched画的红线与绿线数
    "image_matched_r": 20,
    "image_matched_g": 5,

    # 图像拼接时是否判断关键点匹配程度
    "key_points_judge_run": True,
    # 图像融合优化开关
    "image_fusion_run": True,
    # 重叠部分显示比例
    # "p_trans_style": ['average', 'weighting'],
    "p_trans_style": 'weighting',

    # trans递增矩阵形式
    # "trans_style": ['side_to_edge', 'edge_to_center'],
    "trans_style": 'edge_to_center',
    # 生成dis矩阵时的系数，可以加强系数矩阵的非线性
    "power": 3,

    # 创建画布时留下的冗余倍数
    "k": 1,
    # 图像融合腐蚀系数
    "erode": 3,

    # 是否显示图像拼接图片窗口
    "stitch_temp_image_show": True,
    # 是否显示图像拼接canvas窗口
    "stitch_canvas_show": False,
    # 是否显示融合比例图
    "p_trans_show": False
    })


detect_config = ed({

    "top_n": 1,
    "target_size": (224,224),
    "target_obj" : ['book_jacket','monitor', "person","giant_panda",'book_jacket'],    #'desktop_computer''moving_van','racer'
    "threshold_for_resnet": 0.2,
    "threshold_for_iou": 0.3,   # [0,1]
    "resize": True,
    "resize_rate": 1,
    "iou_method":"giou",  # iou giou diou
    "nms_method":"nms",
    "ss_methods": ["fast","quality"],
    "ss_method":0  # 0: fast 1: quality
})


