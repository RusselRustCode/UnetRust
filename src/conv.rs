use std::ops::{AddAssign, SubAssign};
use ndarray::{Array3, Array4, s, Array2};
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
    flat_kernel_size: usize,
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
        let fan_in = kernel_size * kernel_size * input_size.2; // Количество входных связей на один нейрон
        let std = (2.0 / fan_in as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let mut rng = rand::thread_rng();
        

        
        for f in 0..num_filters{
            for kd in 0..input_size.2{
                for kx in 0..kernel_size{
                    for ky in 0..kernel_size{
                        kernels[[f, kx, ky, kd]] = normal.sample(&mut rng);
                    }
                }
            }
        }

        let optim4d = Optim4D::new(optimizer, (num_filters, kernel_size, kernel_size, input_size.2));
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
            optimizer: optim4d,
            flat_kernel_size: kernel_size * kernel_size * input_size.2,
        };
    }

    pub fn to_im2col(&self, input: &Array3<f32>) -> Array2<f32>{
        let (in_h, in_w, in_c) = self.input_size;
        let kernel = self.kernel_size;
        let out_h = self.output_size.0;
        let out_w = self.output_size.1;


        let mut im2col_matrix = Array2::<f32>::zeros((kernel * kernel * in_c, out_h * out_w));


        let mut col_idx = 0;
        for y in 0..out_h{
            for x in 0..out_w{
                for c in 0..in_c{
                    for ky in 0..kernel{
                        for kx in 0..kernel{
                            let val = input[[x + kx, y + ky, c]];
                            let row_idx = ky * kernel * in_c + kx * in_c + c;
                            im2col_matrix[[row_idx, col_idx]] = val;
                        }
                    }
                }
                col_idx += 1;
            }
        }
        return im2col_matrix;
    }



    pub fn forward(&mut self, input: Array3<f32>) -> Array3<f32> {
        self.input_array = input.clone(); // Сохраняем для backward

        let im2col_matrix = self.to_im2col(&input);

        // Размер: num_filters x (kernel_size * kernel_size * channels)
        let kernels_flat = self.kernels.view().into_shape((self.num_filters, self.flat_kernel_size)).unwrap();


        // Результат: num_filters x (out_h * out_w)
        let output_flat = kernels_flat.dot(&im2col_matrix);

        let output_flat_relu = output_flat.mapv(|x| if x >= 0.0 {x} else {0.01 * x});
        // let active_ratio = output_flat_relu.iter().filter(|&&x| x > 0.0).count() as f32 / output_flat.len() as f32;
        // eprintln!("[CONV FORWARD] Active neurons ratio: {:.1}%", active_ratio * 100.0);

        //(out_h, out_w, num_filters)
        let output_3d = output_flat_relu.into_shape((self.output_size.0, self.output_size.1, self.num_filters)).unwrap();

        self.output_array = output_3d.clone();

        output_3d
    }



    fn col2im(&self, grad_matrix: &Array2<f32>) -> Array3<f32> {
        let (in_h, in_w, in_c) = self.input_size;
        let k = self.kernel_size;
        let out_h = self.output_size.0;
        let out_w = self.output_size.1;

        // Создаем массив для градиентов по входу
        let mut grad_input = Array3::<f32>::zeros((in_h, in_w, in_c));

        // Заполняем градиенты
        let mut col_idx = 0; // Индекс столбца в grad_matrix
        for y in 0..out_h {
            for x in 0..out_w {
                for c in 0..in_c {
                    for ky in 0..k {
                        for kx in 0..k {
                            // Получаем градиент для текущего патча
                            let grad_val = grad_matrix[[ky * k * in_c + kx * in_c + c, col_idx]];
                            // Добавляем его в соответствующий пиксель входа
                            grad_input[[x + kx, y + ky, c]] += grad_val;
                        }
                    }
                }
                col_idx += 1;
            }
        }

        grad_input
    }

    // pub fn backward(&mut self, error: Array3<f32>) -> Array3<f32>{
    //     //error[x, y, f] = dL/dout[x, y, f]
    //     //dz/dWf[i, j, c] = X[x + i, y + j, c]
    //     let mut prev_error: Array3<f32> = Array3::<f32>::zeros(self.input_size);
    //     for f in 0..self.output_size.2{
    //         for y in 0..self.output_size.1{
    //             for x in 0..self.output_size.0{
    //                 if self.output_array[[x, y, f]] <= 0.0{ //Смортим чтобы значение было больше нуля, иначе пропускаем, т.к. в backprop производная Relu 0, нет смысла останавливаться
    //                     continue;
    //                 }

    //                 //Вычисляем так же градиенты по входу, т.к.мы знаем, что градиент по входу n слоя равен градиенту выхода n-1 слоя и потом передадим prev_error в backward в качетсве error для n-1 слоя
    //                 //dL/dX = сумма по каналам(dL/dout * dout/dz * dz/dX), dout/dz = 1(производная relu), dL/dout = error, dz/dX = Wf по формуле в forward 
    //                 prev_error.slice_mut(s![x..x+self.kernel_size, y..y+self.kernel_size, ..]).add_assign(&(error[[x, y, f]] * &self.kernels.slice(s![f, .., .., ..])));
    //                 let input_slice = self.input_array.slice(s![x..x+self.kernel_size, y..y+self.kernel_size, ..]);
    //                 //Градиент по весам: dL/dWf[i, j, c] = сумма(dL/dout[x, y, f] * dout[x, y, f]/dz * dz/dW) => сумма(error[x, y, f] * X[x + i, y + j, c])
    //                 //мы знаем, что dout[x, y, f]/dz это производная ReLu, что в свою очерель равна 1 при x > 0 и 0 при x <= 0, поэтому убираем, т.к. эквивалентна 1, 0 мы пропускаем
    //                 //dL/dout[x, y, f] это error, который мы передаем, причем error для n слоя и prev_error для n+1 слоя, а dz/dW это входы X из формулы в forward
    //                 self.kernel_changes.slice_mut(s![f, .., .., ..]).sub_assign(&(error[[x, y, f]] * &input_slice));
    //             }
    //         }
    //     }
    //     return prev_error;
    // }


    pub fn backward(&mut self, error: Array3<f32>) -> Array3<f32> {
        let (out_h, out_w, num_filters) = self.output_size;
        let k = self.kernel_size;
        let in_c = self.input_size.2;
        let flat_size = k * k * in_c;

        let mut error_relu = error.clone();
        if self.output_array.len() > 0 { // Проверка, что output_array был заполнен
            for f in 0..num_filters {
                for y in 0..out_h {
                    for x in 0..out_w {
                        if self.output_array[[x, y, f]] <= 0.0 {
                            error_relu[[x, y, f]] *= 0.01;
                        }
                    }
                }
            }
        }

        // 2. Вычисляем градиенты по весам
        // Для этого нам нужно "развернуть" входные данные и ошибку
        let im2col_matrix = self.to_im2col(&self.input_array); // Размер: (flat_size, out_h * out_w)

        // "Разворачиваем" ошибку в матрицу размером (num_filters, out_h * out_w)
        let error_flat = error_relu.into_shape((num_filters, out_h * out_w)).unwrap();

        // dL/dW = error_flat * im2col_matrix^T
        let grad_weights = error_flat.dot(&im2col_matrix.t()); // Размер: (num_filters, flat_size)
        

        let grad_weights_reshaped = grad_weights.into_shape((num_filters, k, k, in_c)).unwrap();

        // ЛОГИРОВАНИЕ: после вычисления grad_weights
        // let grad_mean = grad_weights_reshaped.mean().unwrap_or(0.0);
        // let grad_std = grad_weights_reshaped.std(0.0);
        // let grad_abs_max = grad_weights_reshaped.fold(0.0, |acc: f32, &x| acc.max(x.abs()));
        // eprintln!("[CONV] Grad stats: mean={:.6e}, std={:.6e}, max={:.6e}", 
        //         grad_mean, grad_std, grad_abs_max);
        
        // let error_mean = error.mean().unwrap_or(0.0);
        // eprintln!("[CONV] Input error: mean={:.6e}", error_mean);


        self.kernel_changes -= &grad_weights_reshaped;

        // dL/dX = kernels^T * error_flat (в формате im2col)
        let kernels_flat = self.kernels.view().into_shape((num_filters, flat_size)).unwrap();
        let grad_input_flat = kernels_flat.t().dot(&error_flat); // Размер: (flat_size, out_h * out_w)

        let grad_input = self.col2im(&grad_input_flat);

        grad_input
    }


    pub fn update(&mut self, minibatch: usize){
     // Сумма градиентов в batch, поэтому делим на размер batch
        self.kernel_changes /= minibatch as f32;
        self.kernels += &self.optimizer.weights_changes(&self.kernel_changes);
        self.kernel_changes = Array4::<f32>::zeros((self.num_filters, self.kernel_size, self.kernel_size, self.input_size.2));
    }

    pub fn zero(&mut self){
        self.kernel_changes = Array4::<f32>::zeros((self.num_filters, self.kernel_size, self.kernel_size, self.input_size.2));
        self.output_array = Array3::<f32>::zeros((self.output_size));
    }
}