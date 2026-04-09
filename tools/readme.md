## 通过网络结构和pth权重导出onnx

### 1 导出full.onnx

    全量导出由于bev_pool_v2的特性，在后续导出trt engine会存在问题，如果要部署到nvidia显卡上跑，还是依赖三段式

```
python3 export_onnx_merged.py
```

### 2 分段导出onnx

```
python3 export_onnx_simple_123.py
```
