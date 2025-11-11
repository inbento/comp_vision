import cv2
import numpy as np

def gaussian_blur(path):
    windows = [
        'Original image',
        'Gaussian Blur', 
        'Gradient Magnitude',
        'Gradient Direction'
    ]
    
    for window in windows:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1000, 900)

    img = cv2.imread(path)
    cv2.imshow('Original image', img)

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_blur = cv2.GaussianBlur(grayscale, (19, 19), 3)
    cv2.imshow('Gaussian Blur', grayscale_blur)


    grad_x = cv2.Sobel(grayscale_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grayscale_blur, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    gradient_direction = np.arctan2(grad_y, grad_x)
    
    gradient_direction_deg = np.degrees(gradient_direction) % 360
    
    magnitude_display = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    direction_display = cv2.normalize(gradient_direction_deg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    cv2.imshow('Gradient Magnitude', magnitude_display)
    cv2.imshow('Gradient Direction', direction_display)
    
    print("===" * 50)
    print(f"Размер матрицы длин градиентов: {gradient_magnitude.shape}")
    print(f"Минимальная длина градиента: {np.min(gradient_magnitude):.2f}")
    print(f"Максимальная длина градиента: {np.max(gradient_magnitude):.2f}")
    print(f"Средняя длина градиента: {np.mean(gradient_magnitude):.2f}")
    print()
    print(f"Размер матрицы углов градиентов: {gradient_direction_deg.shape}")
    print(f"Минимальный угол: {np.min(gradient_direction_deg):.2f}°")
    print(f"Максимальный угол: {np.max(gradient_direction_deg):.2f}°")
    print(f"Средний угол: {np.mean(gradient_direction_deg):.2f}°")
    print("===" * 50)
    
    print("Пример значений для первых 5x5 пикселей:")
    print("Матрица длин градиентов:")
    print(gradient_magnitude[:5, :5])
    print("\nМатрица углов градиентов (градусы):")
    print(gradient_direction_deg[:5, :5])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    gaussian_blur(r"E:/GitHub/comp_vision/lab4/img.jpg")



if __name__ == "__main__":
    main()   