use crate::utils::{TrainImage, TrainingData};
use ndarray::{Array3, s};
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use std::path::{Path, PathBuf};
use std::fs;
use rand::seq::SliceRandom;

// Структура для хранения путей к файлам
pub struct KvasirData {
    pub images: Vec<PathBuf>,
    pub masks: Vec<PathBuf>,
    // Переименовали size -> len, так как у Vec есть метод size(), и Rust путается
    pub len: usize, 
}

impl KvasirData {
    pub fn new(image_dir: &str, mask_dir: &str) -> Self {
        // 1. Исправление: Collect сразу собирает пути, чтобы не было borrow of moved value
        // Мы также сортируем их, чтобы гарантировать парность изображений и масок
        let mut images: Vec<PathBuf> = fs::read_dir(image_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        
        images.sort();

        let mut masks = Vec::new();
        
        for img_path in &images {
            if let Some(file_name) = img_path.file_name() {
                let mask_path = PathBuf::from(mask_dir).join(file_name);
                if mask_path.exists() {
                    masks.push(mask_path);
                }
            }
        }

        // Если масок меньше чем картинок (или не нашлись), берем минимальное число, чтобы не вышло panic при индексации
        let len = images.len().min(masks.len());

        KvasirData {
            images,
            masks,
            len,
        }
    }
}

// Функция для создания TrainingData, совместимой с вашим CNN
pub fn load_kvasir(image_dir: &str, mask_dir: &str, target_size: usize) -> TrainingData {
    let kvasir = KvasirData::new(image_dir, mask_dir);
    
    let mut trn_img = Vec::<TrainImage>::new();
    let mut trn_label = Vec::<usize>::new(); // Исправлено: просто инициализация вектора
    
    // В Kvasir Seg маски это картинки. 
    // Чтобы уместить это в вашу структуру TrainingData (которая ждет label: usize),
    // мы сохраним и картинку и маску как TrainImage::Image (массив пикселей) в одном векторе trn_img.
    // Лейблы (trn_label) здесь не используются в классическом смысле, но структура их требует.
    
    println!("Загрузка {} изображений...", kvasir.len);

    for i in 0..kvasir.len {
        // Загружаем картинку
        let img_path = &kvasir.images[i];
        let img = image::open(img_path)
            .unwrap()
            .resize_exact(target_size as u32, target_size as u32, image::imageops::FilterType::Nearest)
            .to_luma8(); // Превращаем в оттенки серого (1 канал)
        
        let img_data: Vec<f32> = img.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
        let img_arr = Array3::from_shape_vec((target_size, target_size, 1), img_data).unwrap();
        
        trn_img.push(TrainImage::Image(img_arr));
        
        // Загружаем маску
        let mask_path = &kvasir.masks[i];
        let mask_img = image::open(mask_path)
            .unwrap()
            .resize_exact(target_size as u32, target_size as u32, image::imageops::FilterType::Nearest)
            .to_luma8();
        
        // Преобразуем маску в One-Hot (H, W, 2)
        // Класс 0: фон (значение пикселя 0), Класс 1: полип (значение > 0)
        let mut mask_one_hot = Array3::<f32>::zeros((target_size, target_size, 2));
        
        for (idx, pixel) in mask_img.pixels().enumerate() {
            let x = idx % target_size;
            let y = idx / target_size;
            let val = pixel.0[0];
            
            if val > 127 { // Порог для бинарной маски (0-255)
                mask_one_hot[[x, y, 1]] = 1.0; // Полип (класс 1)
            } else {
                mask_one_hot[[x, y, 0]] = 1.0; // Фон (класс 0)
            }
        }
        
        trn_img.push(TrainImage::Image(mask_one_hot));
        trn_label.push(0); // Заглушка, так как для сегментации нужен One-Hot массив, а не число
    }

    let mut classes = std::collections::HashMap::new();
    classes.insert(0, 0); 
    classes.insert(1, 1);

    TrainingData {
        trn_images: trn_img,
        test_images: vec![], 
        trn_labels: trn_label,
        test_labels: vec![],
        rows: target_size,
        cols: target_size,
        classes: classes,
        trn_size: kvasir.len, // Количество пар (картинка+маска)
        test_size: 0,
    }
}

// Вспомогательная функция для получения конкретной пары (Input, Target)
// Мы сохранили данные последовательно: Img0, Mask0, Img1, Mask1...
pub fn get_kvasir_sample(data: &TrainingData, index: usize) -> (Array3<f32>, Array3<f32>) {
    let img_idx = index * 2;
    let mask_idx = index * 2 + 1;
    
    match &data.trn_images[img_idx] {
        TrainImage::Image(img) => match &data.trn_images[mask_idx] {
            TrainImage::Image(mask) => (img.clone(), mask.clone()),
            _ => panic!("Ожидалась маска по индексу {}", mask_idx),
        },
        _ => panic!("Ожидалось изображение по индексу {}", img_idx),
    }
}