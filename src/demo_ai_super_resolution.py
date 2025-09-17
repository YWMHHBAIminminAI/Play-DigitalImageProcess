import cv2
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    # --- 配置输入输出路径 --- 
    # TODO: ！！！重要：你必须修改下面两个路径为你的实际路径！！！
    input_path = Path(r'D:\AIJumpProject\1924_marathon.jpg')  # 替换为你的图片路径
    output_dir = Path(r'D:\AIJumpProject\outputs')
    output_dir.mkdir(exist_ok=True)  # 创建输出目录
    output_path = output_dir / '1924_marathon_4k_enhanced.png'

    # --- 1. 加载图像 ---
    print("🖼️ 正在加载图像...")
    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"无法在路径 {input_path} 找到图像文件，请检查路径是否正确！")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV是BGR顺序，转为RGB

    # --- 2. 初始化Real-ESRGAN模型 ---
    # 选择模型尺度 ('4x' 表示放大4倍)
    scale = 4
    print(f"⚙️ 正在加载Real-ESRGAN {'x'+str(scale)} 模型...")
    
    # 指定模型路径（realesrgan包会自动下载模型到该目录）
    model_path = 'weights/RealESRGAN_x4plus.pth'
    
    # 初始化增强器
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=400,  # 处理大图像时，将其分块以避免显存溢出
        tile_pad=10,
        pre_pad=0,
        half=False if torch.cuda.is_available() else True  # GPU使用float32，CPU使用float16节省内存
    )

    # --- 3. 执行超分辨率! ---
    print("🚀 正在执行超分辨率增强，这可能需要一些时间...")
    try:
        output, _ = upsampler.enhance(img, outscale=scale)  # outscale指定最终放大倍数
    except RuntimeError as e:
        print('⚠️ 显存不足，正在尝试使用更小的tile大小...', e)
        # 如果显存不够，尝试更小的tile值
        upsampler.tile = 200
        output, _ = upsampler.enhance(img, outscale=scale)
    
    # --- 4. 保存并显示结果 ---
    # 保存增强后的图像 (为保持高质量，保存为PNG格式)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  # 改回BGR供OpenCV保存
    cv2.imwrite(str(output_path), output)
    print(f"✅ 增强完成！结果已保存至: {output_path}")

    # --- 5. 可视化对比 ---
    # 为了并排显示，需要将原始图像缩放到相同尺寸
    height, width = output.shape[:2]
    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title(f'Original Image (Resized to {width}x{height})', fontsize=16)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title('Real-ESRGAN x4 Enhanced Image', fontsize=16)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()