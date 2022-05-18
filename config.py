""" config.py """
from easydict import EasyDict as ed

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
    "stitch_style": 'group',
    # 依次拼接顺序
    # "stitch_order": ['Sequence', 'middle_to_both_sides'],
    "stitch_order": 'Sequence',

    # 是否进行柱面变换
    "cylindrical_projection_run": True,
    # 柱面变换是否进行双线性插值
    "bilinear_interpolation": False,

    # 是否进行多波段融合
    "multi_band_blend": True,
    # 多波段融合图像金字塔层数
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

    # 目标检测显不显示图片窗口
    "object_detection_picture_show": False,
    # 目标检测显不显示cmd文字信息
    "object_detection_text_show": False,
    # 目标检测显不显示字体
    "object_detection_search_box_text_show": True,

    # 候选框颜色
    "rect_color": (0, 0, 255),
    # 候选框宽度
    "rect_line_width": 2,

    # 字体颜色
    "text_color": (0, 0, 255),
    # if text_color == (0, 0, 0):
    #     text_color = (1, 1, 1)
    # 最小字体像素
    "text_min_pixel_h": 50,
    "text_min_pixel_w": 100,

    # 确定选择框方法 "search_box_methods": ['sliding_window', 'selective_search'],
    "search_box_method": 'selective_search',

    # SS方法要高质量(True),要高速度(False)
    "ss_search_box_quality": True,

    # 最小候选框长度     "min_search_frame": 224//rate_min_search_frame,
    "rate_min_search_frame": 3,

    # 候选框长宽比
    "search_box_ratios": [0.5, 1, 2],
    # 候选框移动步长(几次完全走掉)
    "search_box_pace": 4,
    # 候选框缩小比例
    "search_box_zoom_rate": 2,

    # 最低可信度
    "min_credibility": 0.8,

    # 候选框文字上下间隔
    "search_box_margin": 2,

    # 交并比高于多少判定为一个物体
    "min_iou": 0.3
})


