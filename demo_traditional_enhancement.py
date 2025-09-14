import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    # 1. 读取你的1924奥运图片
    # img = cv2.imread('your_downloaded_image.jpg')
    # 为立即演示，我们创建一个模拟的模糊图像
    height, width = 300, 400
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)  # 模拟图像
    blurred_img = cv2.GaussianBlur(img, (15, 15), 0)

    # 2. 应用自定义的传统增强算法 (非专利实现)
    # a. 灰度标准化 (基于ITU-R BT.709标准)
    def itu_grayscale(rgb_img):
        # ITU-R BT.709 灰度化系数
        coefficients = np.array([0.2126, 0.7152, 0.0722])
        grayscale = np.dot(rgb_img[..., :3], coefficients)
        return grayscale.astype(np.uint8)

    # b. 自定义对比度拉伸
    def contrast_stretch(grayscale_img):
        min_val = np.min(grayscale_img)
        max_val = np.max(grayscale_img)
        stretched = (grayscale_img - min_val) * (255.0 / (max_val - min_val))
        return stretched.astype(np.uint8)

    gray_img = itu_grayscale(blurred_img)
    enhanced_img = contrast_stretch(gray_img)

    # 3. 可视化结果
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
    plt.title('Original (Blurred) Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img, cmap='gray')
    plt.title('Enhanced Image (Grayscale Stretched)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('enhancement_comparison.png')  # 保存对比图
    plt.show()

if __name__ == '__main__':
    main()
