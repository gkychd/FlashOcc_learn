"""
导出 FlashOCC 的 ONNX 模型
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

    # 改为 TRT 版本
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

    # 1 构建模型结构
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 2 加载预训练权重
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    # 3 将参数移动到GPU显存
    model.cuda()
    # 4 切换推理模式（禁用dropout、批归一化改为推理模式）
    model.eval()

    print(f'模型加载完成: {checkpoint_file}')

    # 获取数据
    for i, data in enumerate(data_loader):
        # 在nuscense 数据集上
        # data['img_inputs'] = [
        # 0: img          (B, 6, 3, 256, 704)  ← 图像
        # 1: sensor2ego  (B, 6, 4, 4)       
        # 2: ego2global  (B, 4, 4)          
        # 3: intrins    (B, 6, 3, 3)       
        # 4: post_rots   (B, 6, 3, 3)       
        # 5: post_trans (B, 6, 3)          
        # 6: bda_matrix (B, 4, 4)         ← 增强变换
        # ]

        # 原始 data['img_inputs'][0]:
        # (B, N_cams, C, H, W) = (1, 6, 3, 256, 704)
        #          ↓ squeeze(0)
        #  img:
        # (6, 3, 256, 704)  = 6个相机视角

        # Step 1: 将 batch 数据移到 GPU
        inputs = [t.cuda() for t in data['img_inputs'][0]]
        # Step 2: 去除 batch 维度 (假设 batch=1)
        # squeeze(0) 移除第0维的1
        # 原来: (1, 6, 3, 256, 704)
        # 现在: (6, 3, 256, 704)
        img = inputs[0].squeeze(0)
        # Step 3: 截取前6个视角 (防止多视角情况)
        if img.shape[0] > 6:
            img = img[:6]

        model_name = model.__class__.__name__
        print(f'模型类型: {model_name}')

        #禁用梯度计算，推理、导出onnx、模型评估时必须使用with torch.no_grad():
        with torch.no_grad():
            # ===== part1: img -> tran_feat, depth =====
            print('\n导出 part1...')
            #替换模型的forward方法，让torch.onnx.export()调用model.forward_part1()
            # # 等价于
            # def new_forward(img):
            #   return model.forward_part1(img)
            # model.forward = new_forward
            model.forward = lambda img: model.forward_part1(img)

            onnx_path = os.path.join(output_dir, 'part1.onnx')
            torch.onnx.export(
                model,
                (img.float().contiguous(),),
                onnx_path,
                export_params=True,
                opset_version=11,
                input_names=['img'],
                output_names=['tran_feat', 'depth']
            )
            print(f'part1 导出完成: {onnx_path}')
            
            # 验证
            onnx_model = onnx.load(onnx_path)
            try:
                onnx.checker.check_model(onnx_model)
                print('ONNX Model Correct')
            except Exception as e:
                print(f'ONNX Model Incorrect: {e}')

            # 运行一次forward获取中间结果
            tran_feat, depth = model.forward_part1(img.float().contiguous())
            print(f'tran_feat shape: {tran_feat.shape}, depth shape: {depth.shape}')

            # 导出带中间维度的onnx
            model_shape_onnx = onnx.load(onnx_path)
            # 自动推导所有层维度！
            model_shape_onnx = shape_inference.infer_shapes(model_shape_onnx)
            # 保存
            onnx_path = os.path.join(output_dir, 'part1_with_shape.onnx')
            onnx.save(model_shape_onnx, onnx_path)
            print(f'part1_with_shape 导出完成: {onnx_path}')

            # ===== 获取 BEV pooling 需要的 ranks 参数 =====
            if model_name in ['BEVDetOCCTRT']:
                metas = model.get_bev_pool_input(inputs)
                ranks_depth = metas[1].int()
                ranks_feat = metas[2].int()
                ranks_bev = metas[0].int()
                interval_starts = metas[3].int()
                interval_lengths = metas[4].int()
                print(f'ranks_depth shape: {ranks_depth.shape}')
                print(f'ranks_feat shape: {ranks_feat.shape}')
                print(f'ranks_bev shape: {ranks_bev.shape}')
                print(f'interval_starts shape: {interval_starts.shape}')
                print(f'interval_lengths shape: {interval_lengths.shape}')

                # ===== part2: tran_feat + depth -> bev_feat =====
                print('\n尝试导出 part2...')
                print(f'depth.numel() = {depth.numel()}, tran_feat.numel() = {tran_feat.numel()}')

                # 尝试直接调用 forward_part2 导出
                try:
                    from functools import partial

                    # 直接绑定 ranks 参数，然后调用
                    # forward_part2默认需要传入7个参数
                    # 使用 partial 先绑定5个参数，这样后续调用时只需要传入2个参数，其余5个参数会自动填充
                    # 这么做的目的是，这些预传入的参数是数据中计算出来的常量（推理时是固定的），不是推理时的输入
                    model.forward = partial(
                        model.forward_part2,
                        ranks_depth=ranks_depth,
                        ranks_feat=ranks_feat,
                        ranks_bev=ranks_bev,
                        interval_starts=interval_starts,
                        interval_lengths=interval_lengths
                    )

                    onnx_path = os.path.join(output_dir, 'part2.onnx')
                    # 这里part2虽然是自定义的实现，但是自定义类继承自torch.autograd.Function，而这个类定义了 symbolic 方法
                    # 因此在转onnx时会识别到 mmdeploy::bev_pooli_v2这个算子
                    # 但是这个节点只有图示意义，运行时需要定义其实现（onnx runtime or tensorrt ...）
                    torch.onnx.export(
                        model,
                        (tran_feat.float().contiguous(), depth.float().contiguous()),
                        onnx_path,
                        export_params=True,
                        opset_version=11,
                        input_names=['tran_feat', 'depth'],
                        output_names=['bev_feat']
                    )
                    print(f'part2 导出完成: {onnx_path}')

                except Exception as e:
                    print(f'part2 导出失败: {e}')

            # 导出带中间维度的onnx
            model_shape_onnx = onnx.load(onnx_path)
            # 自动推导所有层维度！
            model_shape_onnx = shape_inference.infer_shapes(model_shape_onnx)
            # 保存
            onnx_path = os.path.join(output_dir, 'part2_with_shape.onnx')
            onnx.save(model_shape_onnx, onnx_path)
            print(f'part2_with_shape 导出完成: {onnx_path}')

            # ===== forward_part3: bev_feat -> occ_pred =====
            # 直接使用 occ_head 绕过 pts_bbox_head 问题
            print('\n导出 part3...')

            # 自定义 forward，只输出 occupancy
            def forward_part3_occ(x):
                x = x.reshape(1, 200, 200, 64) # reshape不改变地址顺序，只是重新打包
                x = x.permute(0, 3, 1, 2).contiguous() # permute会改变排布顺序，但是地址不连续，通常要调用contiguous()使之真正改变排布
                bev_feature = model.img_bev_encoder_backbone(x)
                occ_bev_feature = model.img_bev_encoder_neck(bev_feature)
                if model.upsample:
                    occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
                outs = model.occ_head(occ_bev_feature)
                return outs

            model.forward = forward_part3_occ

            # bev_feat shape: (1, 64, 200, 200)
            fake_bev_feat = torch.zeros(1, 64, 200, 200).cuda()
            onnx_path = os.path.join(output_dir, 'part3.onnx')
            try:
                torch.onnx.export(
                    model,
                    (fake_bev_feat,),
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    input_names=['bev_feat'],
                    output_names=['occ_pred']
                )
                print(f'part3 导出完成: {onnx_path}')
            except Exception as e:
                print(f'part3 导出失败: {e}')

            # 导出带中间维度的onnx
            model_shape_onnx = onnx.load(onnx_path)
            # 自动推导所有层维度！
            model_shape_onnx = shape_inference.infer_shapes(model_shape_onnx)
            # 保存
            onnx_path = os.path.join(output_dir, 'part3_with_shape.onnx')
            onnx.save(model_shape_onnx, onnx_path)
            print(f'part3_with_shape 导出完成: {onnx_path}')

            print('\n===== 导出完成 =====')
            print(f'输出目录: {output_dir}')
            print('导出了 part1.onnx part2.onnx part3.onnx')
            break


if __name__ == '__main__':
    main()