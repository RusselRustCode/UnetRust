use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use image::io::Reader as ImageReader;

#[derive(Serialize, Deserialize, Clone)]
pub enum TrainImage{
    Image(Array3<f32>),
    Path(PathBuf),
}

#[derive(Serialize, Deserialize, Default)]
pub struct TrainingData{
    pub trn_images: Vec<TrainImage>,
    pub test_images: Vec<TrainImage>,
    pub trn_labels: Vec<usize>, 
    pub test_labels: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    pub classes: HashMap<usize, usize>,
    pub trn_size: usize, 
    pub test_size: usize,
}

#[derive(Serialize, Deserialize)]
pub enum SavingStrategy {
    EveryEpoch(bool),
    EveryNthEpoch(bool, f32),
    BestTrainingAccuracy(bool),
    BestTestingAccuracy(bool),
    Never,
}

pub fn outer(x: Array1<f32>, y: Array1<f32>) -> Array2<f32>{
    let mut result: Array2<f32> = Array2::<f32>::zeros((x.len(), y.len()));

    for i in 0..x.len(){
        for j in 0..y.len(){
            result[[i, j]] = x[i] * y[j];
        }
    }

    return result;
}


pub fn load_img(path: &Path) -> Result<Array3<f32>, String>{
    
    let img = ImageReader::open(path)
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())?;
    let img = img.to_rgb8();

    let rows = img.height() as usize;
    let cols = img.width() as usize;
    let mut array = Array3::zeros((rows, cols, 3));

    for (x, y, pixel) in img.enumerate_pixels() {
        let (r, g, b) = (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
        array[[y as usize, x as usize, 0]] = r / 255.0;
        array[[y as usize, x as usize, 1]] = g / 255.0;
        array[[y as usize, x as usize, 2]] = b / 255.0;
    }

    Ok(array)
}


pub fn load_img_to_grayscale(path: &Path) -> Result<Array3<f32>, String>{
    let img = ImageReader::open(path)
    .map_err(|e| e.to_string())?
    .decode()
    .map_err(|e| e.to_string())?;

    let img_gray = img.to_luma8();
    let (rows, cols) = img_gray.dimensions();
    let mut array = Array3::<f32>::zeros((rows as usize, cols as usize, 1));

    for(x, y, &pixels) in img_gray.enumerate_pixels(){
        let pixel = pixels[0] as f32;
        array[[y as usize, x as usize, 0]] = pixel / 255.0;
    }

    Ok(array)
}
