""" config.py """
from easydict import EasyDict as ed

main_config = ed({
    # 文件名称
    "filename": "book",
    "filename_obj": "src/image/book/result/result.jpg"
})

stitching_config = ed({
    # 图像格式备选
    "image_format_list": ['jpg', 'png', 'bmp', 'tif', 'gif', 'jpeg', ''],
    # 结果的保存格式
    "imwrite_format": 'jpg',

    # 图像压缩大小(一个通道内至多有多少像素)
    "image_size": 1600000,
    # 图像裁剪方式
    # "cut_style_list": ['all', 'reel'],
    "cut_style": 'all',
    # 图像拼接方式：
    # "stitch_style": ['successive', 'group'],
    "stitch_style": 'successive',
    # 依次拼接顺序
    # "stitch_order": ['Sequence', 'middle_to_both_sides'],
    "stitch_order": 'middle_to_both_sides',

    # 是否进行柱面变换
    "cylindrical_projection_run": True,
    # 柱面变换是否进行双线性插值
    "bilinear_interpolation": False,

    # 是否进行多频段融合
    "multi_band_blend": True,
    # 多频段融合图像金字塔层数
    # "pyr_levels": 'max',
    "pyr_levels": 6,

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
    # 生成掩膜矩阵时的系数，可以加强系数矩阵的非线性
    "power": 4,
    # 是否进行掩膜衰减
    "mask_decay": True,
    # 掩膜衰减权重
    "w": 0.95,

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
    # 是否压缩图片
    "resize": True,
    # 图片压缩比例
    "resize_rate": 18,

    # 找候选框方法 可选 ss gradient_mv
    "find_rect_method": "gradient_mv",

    # ss参数
    'scale': 500,
    'sigma': 0.3,
    'min_size': 200,
    "mintargetsize": 300,

    # gmv参数
    # 框宽度
    'w': 110,
    # 框高度
    'h': 60,
    # 最低梯度阈值
    'min_thre': 5000,
    # 最高梯度阈值
    'max_thre': 1000000,
    # 上调比例
    'un': 2,
    # 下调比例
    'dn': 0.5,
    # x, y方向步长
    'x_step': 20,
    'y_step': 20,

    # 选择用什么样的模型 可选 resnet50，resnet152
    "modelsize": "resnet50",
    # resnet检测时，返回前n个可能的目标
    "top_n": 5,
    # resnet检测目标大小
    "target_size": (224, 224),
    # 待检测对象 写入想要检测的类名
    "target_obj": ['mountain_bike', 'moped'],#, minivan，moving_van，racer'],
    # resnet检测最低阈值
    "threshold_for_resnet": 0.4,

    # 非极大值抑制的阈值，应当在 [0,1] 范围内
    "threshold_for_iou": 0.1,
    # iou计算方法 可选择 iou giou diou
    "iou_method": "iou",
    # 非极大值抑制方法 可选择 nms soft_nms
    "nms_method": "nms",

    # 自动调参？
    "autosearch": False,

})


