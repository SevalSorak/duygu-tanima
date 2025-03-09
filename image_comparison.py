import cv2
import os
from skimage.metrics import structural_similarity as ssim

def remove_similar_images(image_folder, threshold=0.99):
    """Belirtilen klasördeki tekrar eden veya çok benzer görselleri siler."""
    images = sorted(os.listdir(image_folder))  
    removed_count = 0  

    for i in range(len(images) - 1):
        img_path1 = os.path.join(image_folder, images[i])
        img_path2 = os.path.join(image_folder, images[i + 1])

        # Görselleri yükle
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Hata: {img_path1} veya {img_path2} yüklenemedi, atlanıyor...")
            continue

        # Görsellerin boyutlarını eşitle
        img1 = cv2.resize(img1, (48, 48))
        img2 = cv2.resize(img2, (48, 48))

        # SSIM hesapla
        similarity = ssim(img1, img2)

        print(f"{images[i]} ve {images[i+1]} SSIM: {similarity:.4f}")

        # Eğer iki görüntü neredeyse aynıysa sil
        if similarity > threshold:
            print(f"📌 {img_path2} siliniyor çünkü {img_path1} ile çok benzer!")
            os.remove(img_path2)
            removed_count += 1

    print(f"🚀 Toplam {removed_count} tekrar eden görsel kaldırıldı.")

dataset_path = r"C:\Users\PC\Downloads\archive\CK+48" 
remove_similar_images(dataset_path, threshold=0.99)
