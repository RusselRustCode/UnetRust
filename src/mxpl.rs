use serde::{Deserialize, Serialize};
use ndarray::{Array3, Array4};
use std::fmt::Debug;

#[derive(Serialize, Deserialize)]
pub struct MaxPooling{
    size: (usize, usize, usize),
    kernel_size: usize,
    #[serde(skip)]
    pub highest_indx: Array4<usize>,
    stride: usize, 
    pub output_size: (usize, usize, usize)
}


impl Debug for MaxPooling{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        s.push_str(&format!("MaxPooling слой\n"));
        s.push_str(&format!("inputs_size: {}-{}-{}\n", self.size.0, self.size.1, self.size.2));
        s.push_str(&format!("размер ядра равен {}\n", self.kernel_size));
        s.push_str(&format!("stride равен {}\n", self.stride));
        s.push_str(&format!("otput size равен {}-{}-{}\n", self.output_size.0, self.output_size.1, self.output_size.2));
        

        return write!(f, "{}", s);
    }
}

impl MaxPooling{
    pub fn new(size: (usize, usize, usize), kernel_size: usize, stride: usize) -> Self{
        let output_widht: usize = ((size.0 - kernel_size) / stride) + 1;
        return Self{
            size: (size.0, size.1, size.2),
            kernel_size: kernel_size,
            highest_indx: Array4::<usize>::zeros((output_widht, output_widht, size.2, 2)),
            output_size: (output_widht, output_widht, size.2),
            stride: stride,
        };
    }

    pub fn zeros(&mut self){
       self.highest_indx = Array4::<usize>::zeros((self.output_size.0, self.output_size.1, self.output_size.2, 2)); 
    }

    pub fn forward(&mut self, input: Array3<f32>) -> Array3<f32>{
        let mut output = Array3::<f32>::zeros(self.output_size);

        for f in 0..self.output_size.2{
            for y in 0..self.output_size.1{
                for x in 0..self.output_size.0{
                    output[[x, y, f]] = -1.0;
                    self.highest_indx[[x, y, f, 0]] = 0;
                    self.highest_indx[[x, y, f, 1]] = 0;

                    for ky in 0..self.kernel_size{
                        for kx in 0..self.kernel_size{
                            let index: (usize, usize) = (x * self.stride + kx, y * self.stride + ky);
                            let value: f32 = input[[index.0, index.1, f]];

                            if value > output[[x, y, f]]{
                                output[[x, y, f]] = value;
                                self.highest_indx[[x, y, f, 0]] = index.0;
                                self.highest_indx[[x, y, f, 1]] = index.1;
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    pub fn backward(&mut self, error: Array3<f32>) -> Array3<f32>{
        let mut prev_error = Array3::<f32>::zeros(self.size);
        
        for f in 0..self.output_size.2{
            for y in 0..self.output_size.1{
                for x in 0..self.output_size.0{
                    let hx = self.highest_indx[[x, y, f, 0]];
                    let hy = self.highest_indx[[x, y, f, 1]];
                    prev_error[[hx, hy, f]] = error[[x, y, f]]; 
                }
            }
        }
        return prev_error;
    }
}