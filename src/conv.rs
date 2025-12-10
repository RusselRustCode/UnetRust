use std::ops::{AddAssign, SubAssign};
use ndarray::{Array3, Array4, s};
use serde::{Serialize, Deserialize};
use std::fmt::{Debug, Formatter};
use rand_distr::{Normal, Distribution};
use crate::optim::{Optim4D, Optimizer};

#[derive(Serialize, Deserialize)]
pub struct ConvolutionLayer{
    input_size: (usize, usize, usize),
    kernel_size: usize,
    pub output_size: (usize, usize, usize),
    #[serde(skip)]
    input_array: Array3<f32>,
    #[serde(skip)]
    output_array: Array3<f32>,
    stride: usize,
    padding: usize,
    num_filters: usize,
    kernels: Array4<f32>,
    #[serde(skip)]
    kernel_changes: Array4<f32>,
    optimizer: Optim4D,
}

impl Debug for ConvolutionLayer{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        s.push_str("Сверточый слой");
        s.push_str(&format!("Размер Входа: {}x{}x{}\n", self.input_size.0, self.input_size.1, self.input_size.2));
        s.push_str(&format!("Размер ядра: {}x{}\n", self.kernel_size, self.kernel_size));
        s.push_str(&format!("Размер Выхода: {}x{}x{}\n", self.output_size.0, self.output_size.1, self.output_size.2));
        s.push_str(&format!("Stride: {}\n", self.stride));
        s.push_str(&format!("Количество фильтров(признаков): {}\n", self.num_filters));

        write!(f, "{}", s)
    }
}

impl ConvolutionLayer{
    pub fn new(input_size: (usize, usize, usize), kernel_size: usize, stride: usize, padding: Option<usize>, num_filters: usize, optimizer: Optimizer) -> Self{
        let output = ((input_size.0 - kernel_size) / stride) + 1;
        let oputut_size = (output, output, num_filters);
        let mut kernels = Array4::zeros((num_filters, kernel_size, kernel_size, input_size.2));
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();
        for f in 0..num_filters{
            for kd in 0..input_size.2{
                for kx in 0..kernel_size{
                    for ky in 0..kernel_size{
                        kernels[[f, kx, ky, kd]] = normal.sample(&mut rng) * (2.0 / (input_size.0.pow(2)) as f32).sqrt()
                    }
                }
            }
        }

        let optim4d = Optim4D::new(optimizer, (num_filters, kernel_size, kernel_size, input_size.0));
        let padding = match padding {
            Some(padding) => padding,
            None => 0,
        };

        return Self{
            input_size: input_size,
            kernel_size: kernel_size,
            output_size: oputut_size,
            input_array: Array3::zeros(input_size),
            output_array: Array3::zeros(oputut_size),
            stride: stride, 
            padding: padding,
            num_filters: num_filters,
            kernels: kernels,
            kernel_changes: Array4::zeros((num_filters, kernel_size, kernel_size, input_size.2)),
            optimizer: optim4d
        };
    }


    pub fn forward(&mut self, input: Array3<f32>) -> Array3<f32>{
        self.input_array = input;
        for f in 0..self.output_size.2{
            let kernel_slice = self.kernels.slice(s![f, .., .., ..]);
            for y in 0..self.output_size.1{
                for x in 0..self.output_size.0{
                    let input_slice = self.input_array.slice(s![x..x+self.kernel_size, y..y+self.kernel_size, ..]);
                    self.output_array[[x, y, f]] = (&input_slice * &kernel_slice).sum().max(0.0);
                }
            }
        }
        
        self.output_array.clone()
    } 


    pub fn backward(&mut self, error: Array3<f32>) -> Array3<f32>{
        let mut prev_error: Array3<f32> = Array3::<f32>::zeros(self.input_size);
        for f in 0..self.output_size.2{
            for y in 0..self.output_size.1{
                for x in 0..self.output_size.0{
                    if self.output_array[[x, y, f]] <= 0.0{
                        continue;
                    }

                    prev_error.slice_mut(s![x..x+self.kernel_size, y..y+self.kernel_size, ..]).add_assign(&(error[[x, y, f]] * &self.kernels.slice(s![f, .., .., ..])));
                    let input_slice = self.input_array.slice(s![x..x+self.kernel_size, y..y+self.kernel_size, ..]);
                    self.kernel_changes.slice_mut(s![f, .., .., ..]).sub_assign(&(error[[x, y, f]] * &input_slice));
                }
            }
        }
        return prev_error;
    }

    pub fn update(&mut self, minibatch: usize){
        self.kernel_changes /= minibatch as f32;
        self.kernels += &self.optimizer.weights_changes(&self.kernel_changes);
        self.kernel_changes = Array4::<f32>::zeros((self.num_filters, self.kernel_size, self.kernel_size, self.input_size.2));
    }

    pub fn zero(&mut self){
        self.kernel_changes = Array4::<f32>::zeros((self.num_filters, self.kernel_size, self.kernel_size, self.input_size.2));
        self.output_array = Array3::<f32>::zeros((self.output_size));
    }
}