"""
导出 FlashOCC 的 ONNX 模型
"""
import os
import sys
sys.path.insert(0, os.getcwd())

from functools import partial

import torch
import onnx
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor


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

    # 导入 plugin 模块以注册数据集
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

    # 构建数据加载器
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        test_dl = getattr(cfg.data, 'test_dataloader', None)
        if test_dl and test_dl.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        test_dl = getattr(cfg.data, 'test_dataloader', None)
        if test_dl and test_dl.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {**test_dataloader_default_args}
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # 构建模型并加载权重
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

        # 根据模型类型获取 bev_pool 输入
        model_name = model.__class__.__name__
        print(f'模型类型: {model_name}')

        if model_name in ['FBOCCTRT', 'FBOCC2DTRT', 'BEVDetOCCTRT']:
            if model_name == 'BEVDetOCCTRT':
                metas = model.get_bev_pool_input(inputs)
            else:
                metas = model.get_bev_pool_input(inputs, img_metas=data['img_metas'])
            onnx_input = (
                img.float().contiguous(),
                metas[1].int().contiguous(),
                metas[2].int().contiguous(),
                metas[0].int().contiguous(),
                metas[3].int().contiguous(),
                metas[4].int().contiguous()
            )
            input_names = [
                'img', 'ranks_depth', 'ranks_feat', 'ranks_bev',
                'interval_starts', 'interval_lengths'
            ]
        elif model_name == 'BEVDetOCC':
            # BEVDetOCC 直接导出，仅接受图像输入
            onnx_input = (img.float().contiguous(),)
            input_names = ['img']
        else:
            raise ValueError(f'不支持的模型类型: {model_name}')

        dynamic_axes = {
            "ranks_depth": {0: 'M'},
            "ranks_feat": {0: 'M'},
            "ranks_bev": {0: 'M'},
            "interval_starts": {0: 'N'},
            "interval_lengths": {0: 'N'},
        }

        # 确定输出 - BEVDetOCC 只有 occ_head
        if model.pts_bbox_head is not None and hasattr(model, 'wdet3d') and model.wdet3d:
            if hasattr(model, 'wocc') and model.wocc:
                output_names = [f'output_{j}' for j in range(1 + 6 * len(model.pts_bbox_head.task_heads))]
            else:
                output_names = [f'output_{j}' for j in range(6 * len(model.pts_bbox_head.task_heads))]
        else:
            # BEVDetOCC 只有 occupancy head
            output_names = ['occ_pred']

        # 导出完整模型
        output_file = os.path.join(output_dir, 'flashocc-r50.onnx')

        # 使用 forward_dummy 方法，避免 return_loss 参数问题
        model.forward = model.forward_dummy

        with torch.no_grad():
            torch.onnx.export(
                model,
                onnx_input,
                output_file,
                export_params=True,
                opset_version=11,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )

        # 验证 ONNX 模型
        onnx_model = onnx.load(output_file)
        try:
            onnx.checker.check_model(onnx_model)
            print('ONNX 模型验证通过')
        except Exception as e:
            print(f'ONNX 模型验证警告: {e}')

        # 检查模型是否有 forward_with_argmax 方法
        if hasattr(model, 'forward_with_argmax'):
            # 导出带 argmax 的版本
            model.forward = model.forward_with_argmax
            output_file_argmax = os.path.join(output_dir, 'flashocc-r50_with_argmax.onnx')

            with torch.no_grad():
                torch.onnx.export(
                    model,
                    onnx_input,
                    output_file_argmax,
                    export_params=True,
                    opset_version=11,
                    input_names=input_names,
                    output_names=['cls_occ_label'],
                    dynamic_axes=dynamic_axes
                )

            onnx_model = onnx.load(output_file_argmax)
            try:
                onnx.checker.check_model(onnx_model)
                print('ONNX (with_argmax) 模型验证通过')
            except Exception as e:
                print(f'ONNX (with_argmax) 模型验证警告: {e}')

            print(f'\n导出完成!')
            print(f'  - {output_file}')
            print(f'  - {output_file_argmax}')
        else:
            print(f'\n导出完成! (模型不支持 forward_with_argmax)')
            print(f'  - {output_file}')
        break


if __name__ == '__main__':
    main()