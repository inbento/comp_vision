import cv2
import numpy as np

def setup_windows(window_names):
    for window in window_names:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1000, 900)

def load_and_preprocess_image(path):

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (19, 19), 3)
    
    return img, blurred

def compute_gradients(blurred_image):

    grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    direction_deg = np.degrees(direction) % 360
    
    return grad_x, grad_y, magnitude, direction_deg

def non_maximum_suppression(magnitude, direction):

    M, N = magnitude.shape
    suppressed = np.zeros((M, N), dtype=np.float64)
    
    direction_deg = direction % 180
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            angle = direction_deg[i, j]
            mag = magnitude[i, j]
            
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            else:
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            
            if mag >= max(neighbors):
                suppressed[i, j] = mag
    
    return suppressed

def double_threshold_filter(suppressed_magnitude, low_ratio=0.1, high_ratio=0.3):

    high_threshold = np.max(suppressed_magnitude) * high_ratio
    low_threshold = high_threshold * low_ratio
    
    strong_edges = np.zeros_like(suppressed_magnitude, dtype=np.uint8)
    weak_edges = np.zeros_like(suppressed_magnitude, dtype=np.uint8)
    
    strong_i, strong_j = np.where(suppressed_magnitude >= high_threshold)
    weak_i, weak_j = np.where((suppressed_magnitude >= low_threshold) & 
                             (suppressed_magnitude < high_threshold))
    
    strong_edges[strong_i, strong_j] = 255
    weak_edges[weak_i, weak_j] = 255
    
    return strong_edges, weak_edges, low_threshold, high_threshold

def apply_hysteresis(strong_edges, weak_edges):

    final_edges = strong_edges.copy()
    M, N = strong_edges.shape
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            if weak_edges[i, j] == 255:
                if np.any(strong_edges[i-1:i+2, j-1:j+2] == 255):
                    final_edges[i, j] = 255
    
    return final_edges

def display_images(images_dict):

    for window_name, image in images_dict.items():
        cv2.imshow(window_name, image)

def print_analysis(gradient_magnitude, suppressed_magnitude, gradient_direction_deg, 
                  strong_edges, weak_edges, final_edges, low_threshold, high_threshold):

    print("=" * 50)
    print("АНАЛИЗ ДО ПОДАВЛЕНИЯ НЕМАКСИМУМОВ:")
    print(f"Размер матрицы длин градиентов: {gradient_magnitude.shape}")
    print(f"Минимальная длина градиента: {np.min(gradient_magnitude):.2f}")
    print(f"Максимальная длина градиента: {np.max(gradient_magnitude):.2f}")
    print(f"Средняя длина градиента: {np.mean(gradient_magnitude):.2f}")
    print(f"Количество ненулевых градиентов: {np.count_nonzero(gradient_magnitude)}")

    print("\n" + "=" * 50)
    print("АНАЛИЗ ПОСЛЕ ПОДАВЛЕНИЯ НЕМАКСИМУМОВ:")
    print(f"Минимальная длина градиента: {np.min(suppressed_magnitude):.2f}")
    print(f"Максимальная длина градиента: {np.max(suppressed_magnitude):.2f}")
    print(f"Средняя длина градиента: {np.mean(suppressed_magnitude):.2f}")
    print(f"Количество ненулевых градиентов: {np.count_nonzero(suppressed_magnitude)}")
    print(f"Процент сохраненных градиентов: {np.count_nonzero(suppressed_magnitude) / np.count_nonzero(gradient_magnitude) * 100:.2f}%")

    print("\n" + "=" * 50)
    print("ДВОЙНАЯ ПОРОГОВАЯ ФИЛЬТРАЦИЯ:")
    print(f"Высокий порог: {high_threshold:.2f}")
    print(f"Низкий порог: {low_threshold:.2f}")
    print(f"Количество сильных границ: {np.count_nonzero(strong_edges)}")
    print(f"Количество слабых границ: {np.count_nonzero(weak_edges)}")
    print(f"Количество финальных границ: {np.count_nonzero(final_edges)}")
    print(f"Процент сильных границ от подавленных: {np.count_nonzero(strong_edges) / np.count_nonzero(suppressed_magnitude) * 100:.2f}%")
    print(f"Процент финальных границ от подавленных: {np.count_nonzero(final_edges) / np.count_nonzero(suppressed_magnitude) * 100:.2f}%")

    print("\n" + "=" * 50)
    print("МАТРИЦЫ УГЛОВ ГРАДИЕНТОВ:")
    print(f"Размер матрицы углов градиентов: {gradient_direction_deg.shape}")
    print(f"Минимальный угол: {np.min(gradient_direction_deg):.2f}°")
    print(f"Максимальный угол: {np.max(gradient_direction_deg):.2f}°")
    print(f"Средний угол: {np.mean(gradient_direction_deg):.2f}°")

    print("\n" + "=" * 50)
    print("ПРИМЕРЫ ЗНАЧЕНИЙ ДЛЯ ПЕРВЫХ 5x5 ПИКСЕЛЕЙ:")
    print("Матрица длин градиентов (до подавления):")
    print(gradient_magnitude[:5, :5].round(2))
    print("\nМатрица длин градиентов (после подавления):")
    print(suppressed_magnitude[:5, :5].round(2))
    print("\nМатрица углов градиентов (градусы):")
    print(gradient_direction_deg[:5, :5].round(2))
    print("=" * 50)

def full_canny(path):
    window_names = [
        'Original image',
        'Gaussian Blur', 
        'Gradient Magnitude',
        'Gradient Direction',
        'Non-Maximum Suppression',
        'Final Edges'
    ]
    setup_windows(window_names)
    
    original, blurred = load_and_preprocess_image(path)
    
    _, _, magnitude, direction_deg = compute_gradients(blurred)
    
    suppressed = non_maximum_suppression(magnitude, direction_deg)
    
    strong_edges, weak_edges, low_th, high_th = double_threshold_filter(suppressed)
    final_edges = apply_hysteresis(strong_edges, weak_edges)
    
    magnitude_display = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    direction_display = cv2.normalize(direction_deg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    suppressed_display = cv2.normalize(suppressed, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    images_to_display = {
        'Original image': original,
        'Gaussian Blur': blurred,
        'Gradient Magnitude': magnitude_display,
        'Gradient Direction': direction_display,
        'Non-Maximum Suppression': suppressed_display,
        'Final Edges': final_edges
    }
    
    display_images(images_to_display)
    
    print_analysis(magnitude, suppressed, direction_deg, strong_edges, 
                    weak_edges, final_edges, low_th, high_th)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return final_edges
        

def main():
    image_path = r"E:/GitHub/comp_vision/lab4/img.jpg"
    full_canny(image_path)

if __name__ == "__main__":
    main()