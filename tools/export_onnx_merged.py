"""
导出 FlashOCC 完整 ONNX 模型 (合并版)
"""
import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import onnx
from onnx import shape_inference
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def main():
    # 配置路径
    config_file = 'projects/configs/flashocc/flashocc-r50.py'
    checkpoint_file = 'work_dirs/flashocc-r50/epoch_24_ema.pth'
    output_dir = 'work_dirs/export_onnx'

    os.makedirs(output_dir, exist_ok=True)

    # 加载配置
    cfg = Config.fromfile(config_file)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.type = cfg.model.type + 'TRT'

    try:
        from mmdet.utils import compat_cfg
    except ImportError:
        from mmdet3d.utils import compat_cfg
    cfg = compat_cfg(cfg)

    # 导入 plugin
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
        else:
            plugin_dir = os.path.dirname(config_file)
        _module_dir = plugin_dir.strip('/').split('/')
        _module_path = _module_dir[0]
        for m in _module_dir[1:]:
            _module_path = _module_path + '.' + m
        print(f'Loading plugin: {_module_path}')
        plg_lib = importlib.import_module(_module_path)

    # 数据加载器
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    test_loader_cfg = {**test_dataloader_default_args}
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # 构建模型
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    model.cuda()
    model.eval()

    print(f'模型加载完成: {checkpoint_file}')

    # 获取数据
    for i, data in enumerate(data_loader):
        inputs = [t.cuda() for t in data['img_inputs'][0]]
        img = inputs[0].squeeze(0)
        if img.shape[0] > 6:
            img = img[:6]

        model_name = model.__class__.__name__
        print(f'模型类型: {model_name}')

        with torch.no_grad():
            if model_name in ['BEVDetOCCTRT']:
                # 获取 BEV pooling 需要的 ranks 参数
                metas = model.get_bev_pool_input(inputs)
                ranks_depth = metas[1].int()
                ranks_feat = metas[2].int()
                ranks_bev = metas[0].int()
                interval_starts = metas[3].int()
                interval_lengths = metas[4].int()
                # 定义完整的 forward (串联3个part)
                def forward_full(img):
                    # ===== part1: img -> tran_feat, depth =====
                    x = model.img_backbone(img)
                    x = model.img_neck(x)
                    x = model.img_view_transformer.depth_net(x[0])
                    depth = x[:, :model.img_view_transformer.D].softmax(dim=1)
                    tran_feat = x[:, model.img_view_transformer.D:(
                        model.img_view_transformer.D +
                        model.img_view_transformer.out_channels)]
                    tran_feat = tran_feat.permute(0, 2, 3, 1)

                    # ===== part2: tran_feat + depth -> bev_feat =====
                    B, D, H, W = 6, 16, 44, 88
                    tran_feat_reshaped = tran_feat.reshape(B, D, H, -1).permute(0, 1, 3, 2).reshape(B, 16, 64, 44)
                    depth_reshaped = depth.reshape(B, D, H, W)
                    x = model.forward_part2(
                        tran_feat_reshaped, depth_reshaped,
                        ranks_depth=ranks_depth,
                        ranks_feat=ranks_feat,
                        ranks_bev=ranks_bev,
                        interval_starts=interval_starts,
                        interval_lengths=interval_lengths
                    )

                    # ===== part3: bev_feat -> occ_pred =====
                    # 这里注意，由于part2的实现，onnx无法识别，因此无法确定part3的实际输入维度，因此part3部分的维度显示上都是未知
                    x = x.reshape(1, 200, 200, 64)
                    x = x.permute(0, 3, 1, 2).contiguous()
                    bev_feature = model.img_bev_encoder_backbone(x)
                    occ_bev_feature = model.img_bev_encoder_neck(bev_feature)
                    if model.upsample:
                        occ_bev_feature = torch.nn.functional.interpolate(
                            occ_bev_feature, scale_factor=2, mode='bilinear', align_corners=True)
                    outs = model.occ_head(occ_bev_feature)

                    return outs

                print('\n导出完整模型...')
                model.forward = forward_full

                onnx_path = os.path.join(output_dir, 'flashocc_full.onnx')
                torch.onnx.export(
                    model,
                    (img.float().contiguous(),),
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    input_names=['img'],
                    output_names=['occ_pred'],
                    verbose=True  # 打印详细信息
                )
                print(f'完整模型导出完成: {onnx_path}')

                # 验证
                onnx_model = onnx.load(onnx_path)
                try:
                    onnx.checker.check_model(onnx_model)
                    print('ONNX Model Correct')
                except Exception as e:
                    print(f'ONNX Model Incorrect: {e}')

                # 打印输入输出维度
                print("\n========== ONNX 模型结构 ==========")
                print(f"输入: img {img.shape}")
                print(f"输出: occ_pred")

                # 打印中间层
                print("\n中间层 (部分节点):")
                for node in onnx_model.graph.node[:20]:  # 只打印前20个节点
                    print(f"  {node.op_type}: {node.name}")

                break

    print('\n===== 导出完成 =====')
    print(f'输出文件: {output_dir}/flashocc_full.onnx')

    # 导出带中间维度的onnx
    model_shape_onnx = onnx.load(onnx_path)
    # 自动推导所有层维度！
    model_shape_onnx = shape_inference.infer_shapes(model_shape_onnx)
    # 保存
    onnx_path = os.path.join(output_dir, 'flashocc_full_with_shape.onnx')
    onnx.save(model_shape_onnx, onnx_path)
    print(f'flashocc_full_with_shape 导出完成: {onnx_path}')

if __name__ == '__main__':
    main()