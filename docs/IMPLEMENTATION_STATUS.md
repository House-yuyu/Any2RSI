# 实现范围

| 论文模块 | 当前实现 |
|---|---|
| EDG | InternVL2.5、Qwen2.5-VL、OpenAI-compatible API、Echo；支持断点恢复、失败记录和 enriched JSON 输出 |
| CMMCA Eq. (3)-(5) | active-control mask、四层 cross-modal aggregation、多控制交叉注意力和四级 query grid |
| 多层 visual tokens | CLIP ViT-L/14 第 6、12、18、24 层特征 |
| 空间控制注入 | AnyControl LocalAdapter 与 UNet 13 级 residual 注入 |
| PGO Eq. (2) | predicted-x0 可微 VAE 解码与 CLIP image-text loss |
| 控制预处理 | Canny、HED、EntitySeg 及预计算控制图缓存 |
| 任意控制组合 | 随机控制组合训练、active mask、单图与批量 DDIM 推理 |
| 训练管理 | warm start、完整 Lightning 续训、adapter 加载与 trainable-only 导出 |
| 评测 | CLIP similarity、PSNR、SSIM 与 clean-FID |
| 工程质量 | 权重 SHA256、环境检查、CPU 单元测试、release hygiene 和 GitHub Actions |

完整数据流见 `docs/ARCHITECTURE.md`，具体参数化选择见
`docs/IMPLEMENTATION_DETAILS.md`，运行记录见 `docs/TRAINING_VALIDATION.md`。
