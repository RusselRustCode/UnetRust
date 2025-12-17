use std::{collections::HashMap, path::Path};
use rust_mnist::Mnist;
use crate::utils::{TrainImage, TrainingData, load_img};
use ndarray::Array3;
use rand::seq::IteratorRandom;
use std::fs::create_dir;

pub fn load_mnist<T>(mnist_path: T) -> TrainingData
where T: AsRef<Path>{
    let (rows, cols) = (28, 28);
    let path = mnist_path.as_ref();
    
    let mnist = Mnist::new(path.to_str().unwrap());

    let mut trn_img = Vec::<TrainImage>::new();
    let mut trn_label = Vec::<usize>::new();
    let mut test_img = Vec::<TrainImage>::new();
    let mut test_label = Vec::<usize>::new();

    let mnist_path = path.join("unpacked");
    if !mnist_path.exists(){
        create_dir(mnist_path.as_path()).expect("Failed");
    }
    let mean = 0.1307; // Среднее значение для MNIST
    let std = 0.3081; 

    for i in 0..60000{
        let flat_data: Vec<f32> = mnist.train_data[i]
        .iter()
        .map(|x| (*x as f32 - mean) / std)
        .collect();

        let img = Array3::from_shape_vec((rows, cols, 1), flat_data).unwrap();

        trn_img.push(TrainImage::Image(img));
        trn_label.push(mnist.train_labels[i] as usize);
    }

    for i in 0..10000{
        let flat_data: Vec<f32> = mnist.test_data[i]
        .iter()
        .map(|x| (*x as f32 - mean) / std)
        .collect();

        let img = Array3::from_shape_vec((rows, cols, 1), flat_data).unwrap();

        test_img.push(TrainImage::Image(img));
        test_label.push(mnist.test_labels[i] as usize);

    }

    let classes: HashMap<usize, usize> = (0..10).enumerate().collect();


    let data: TrainingData = TrainingData{
        trn_images: trn_img,
        test_images: test_img,
        trn_labels: trn_label,
        test_labels: test_label,
        rows: rows,
        cols: cols,
        classes: classes,
        trn_size: 60000,
        test_size: 10000,
    };

    return data;
}


pub fn get_rand_train_image(data: &TrainingData) -> (Array3<f32>, usize){
    let mut rng = rand::thread_rng();

    let (img, label) = data.trn_images.iter().zip(data.trn_labels.iter()).choose(&mut rng).unwrap();

    match img{
        TrainImage::Image(img) => (img.clone(), *label),
        TrainImage::Path(img_path) => {
            let img = load_img(img_path).unwrap();
            return (img, *label);
        },
    }
    
}

pub fn get_rand_test_image(data: &TrainingData) -> (Array3<f32>, usize){
    let mut rng = rand::thread_rng();

    let (img, label) = data.test_images.iter().zip(data.test_labels.iter()).choose(&mut rng).unwrap();

    match img{
        TrainImage::Image(img) => (img.clone(), *label),
        TrainImage::Path(img_path) => {
            let img = load_img(img_path).unwrap();
            return (img, *label);
        },
    }
    
}