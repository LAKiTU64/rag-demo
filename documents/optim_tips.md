# Qwen 优化建议
当 batch_size > 8 时，L2 缓存命中率显著下降。
建议 input_len 控制在 512 以内以避免显存溢出。
热点 kernel: flash_attn_fwd, rms_norm_kernel
对于 qwen-1.8b，推荐 batch_size=1~4。