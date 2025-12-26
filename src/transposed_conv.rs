use ndarray::{Array3, Array4, Array2, s};
use rand_distr::{Normal, Distribution};
use serde::{Serialize, Deserialize};
use std::fmt::Debug;
use crate::optim::{Optim4D, Optimizer};

#[derive(Serialize, Deserialize)]
pub struct TranposeConv {
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
    kernels: Array4<f32>, // (num_filters, k, k, in_channels)
    #[serde(skip)]
    kernel_changes: Array4<f32>,
    optimizer: Optim4D,
    flat_kernel_size: usize, // k * k * in_channels
}

impl Debug for TranposeConv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        s.push_str("Transpose Convolution Layer\n");
        s.push_str(&format!("Input Size: {}x{}x{}\n", self.input_size.0, self.input_size.1, self.input_size.2));
        s.push_str(&format!("Kernel Size: {}x{}\n", self.kernel_size, self.kernel_size));
        s.push_str(&format!("Output Size: {}x{}x{}\n", self.output_size.0, self.output_size.1, self.output_size.2));
        s.push_str(&format!("Stride: {}\n", self.stride));
        s.push_str(&format!("Num Filters: {}\n", self.num_filters));
        write!(f, "{}", s)
    }
}

impl TranposeConv {
    pub fn new(
        input_size: (usize, usize, usize),
        kernel_size: usize,
        stride: usize,
        padding: Option<usize>,
        num_filters: usize,
        optimizer: Optimizer,
    ) -> Self {
        let (w_in, h_in, in_c) = input_size;
        let p = padding.unwrap_or(0);

        // Формула для Transpose Conv
        let out_w = (w_in - 1) * stride + kernel_size - 2 * p;
        let out_h = (h_in - 1) * stride + kernel_size - 2 * p;
        let output_size = (out_w, out_h, num_filters);

        let mut kernels = Array4::zeros((num_filters, kernel_size, kernel_size, in_c));
        // Kaiming init
        let fan_in = kernel_size * kernel_size * in_c;
        let std = (2.0 / fan_in as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let mut rng = rand::thread_rng();

        for f in 0..num_filters {
            for c in 0..in_c {
                for y in 0..kernel_size {
                    for x in 0..kernel_size {
                        kernels[[f, x, y, c]] = normal.sample(&mut rng);
                    }
                }
            }
        }

        let optim4d = Optim4D::new(optimizer, (num_filters, kernel_size, kernel_size, in_c));

        Self {
            input_size,
            kernel_size,
            output_size,
            input_array: Array3::zeros(input_size),
            output_array: Array3::zeros(output_size),
            stride,
            padding: p,
            num_filters,
            kernels,
            kernel_changes: Array4::zeros((num_filters, kernel_size, kernel_size, in_c)),
            optimizer: optim4d,
            flat_kernel_size: kernel_size * kernel_size * in_c,
        }
    }

    /// Подготавливает входные данные как для обычной свертки, но используя размер ВЫХОДА транспонированной.
    /// Это "разреженная" матрица, где каждый столбец соответствует пикселю на ВЫХОДЕ.
    /// В каждом столбце мы собираем значения из ВХОДА, которые влияют на этот пиксель выхода.
    fn im2col_transpose(&self, input: &Array3<f32>) -> Array2<f32> {
        let (in_w, in_h, in_c) = self.input_size;
        let (out_w, out_h, _) = self.output_size;
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;

        // Строки: размер ядра * каналы входа
        // Столбцы: все пиксели выхода
        let mut im2col_matrix = Array2::<f32>::zeros((k * k * in_c, out_w * out_h));

        let mut col_idx = 0;
        for y_out in 0..out_h {
            for x_out in 0..out_w {
                // Рассчитываем центральный пиксель на входе, который влияет на этот участок выхода
                // x_in = (x_out + p) / s - для обычной конволюции (откуда пришли данные)
                // Здесь мы делаем наоборот: ищем, какие входы попадут в x_out
                
                // Формула связи координат:
                // x_out = x_in * s - p + kx
                // => x_in = (x_out + p - kx) / s
                
                for c in 0..in_c {
                    for ky in 0..k {
                        for kx in 0..k {
                            let x_in_float = (x_out + p) as f32 - kx as f32;
                            let y_in_float = (y_out + p) as f32 - ky as f32;

                            // Проверяем, чтобы координаты были целыми и попадали в диапазон входа
                            if x_in_float >= 0.0 && y_in_float >= 0.0 {
                                let x_in = x_in_float as usize;
                                let y_in = y_in_float as usize;
                                
                                // Важная проверка кратности stride
                                if x_in % s == 0 && y_in % s == 0 {
                                    let x_in_src = x_in / s;
                                    let y_in_src = y_in / s;

                                    if x_in_src < in_w && y_in_src < in_h {
                                        let row_idx = ky * k * in_c + kx * in_c + c;
                                        im2col_matrix[[row_idx, col_idx]] = input[[x_in_src, y_in_src, c]];
                                    }
                                }
                            }
                        }
                    }
                }
                col_idx += 1;
            }
        }
        im2col_matrix
    }

    /// Обратная операция к im2col_transpose (для вычисления dInput)
    fn col2im_transpose(&self, grad_matrix: &Array2<f32>) -> Array3<f32> {
        let (in_w, in_h, in_c) = self.input_size;
        let (out_w, out_h, _) = self.output_size;
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;

        let mut grad_input = Array3::<f32>::zeros((in_w, in_h, in_c));

        let mut col_idx = 0;
        for y_out in 0..out_h {
            for x_out in 0..out_w {
                for c in 0..in_c {
                    for ky in 0..k {
                        for kx in 0..k {
                            let row_idx = ky * k * in_c + kx * in_c + c;
                            let grad_val = grad_matrix[[row_idx, col_idx]];

                            let x_in_float = (x_out + p) as f32 - kx as f32;
                            let y_in_float = (y_out + p) as f32 - ky as f32;

                            if x_in_float >= 0.0 && y_in_float >= 0.0 {
                                let x_in = x_in_float as usize;
                                let y_in = y_in_float as usize;
                                
                                if x_in % s == 0 && y_in % s == 0 {
                                    let x_in_dst = x_in / s;
                                    let y_in_dst = y_in / s;

                                    if x_in_dst < in_w && y_in_dst < in_h {
                                        grad_input[[x_in_dst, y_in_dst, c]] += grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
                col_idx += 1;
            }
        }
        grad_input
    }

    pub fn forward(&mut self, input: Array3<f32>) -> Array3<f32> {
        self.input_array = input.clone();

        // 1. Разворачиваем вход (используем логику Transpose)
        let im2col_matrix = self.im2col_transpose(&input); // (flat_k, out_w * out_h)

        // 2. Разворачиваем ядра (NumFilters, flat_k)
        let kernels_flat = self.kernels.view().into_shape((self.num_filters, self.flat_kernel_size)).unwrap();

        // 3. Матричное умножение: Output = Kernels.T * Im2col
        // В ndarray: это (flat_k x num_filters).t() * (flat_k x n_pixels) = (num_filters x n_pixels)
        // Или через kernels_flat.t().dot(...)
        // Но так как kernels_flat (num_filters, flat_k), нам нужно умножить её Транспонированную на im2col_matrix
        // Result shape: (num_filters, out_pixels)
        let output_flat = kernels_flat.t().dot(&im2col_matrix);

        // 4. Leaky ReLU
        let output_flat_relu = output_flat.mapv(|x| if x >= 0.0 { x } else { 0.01 * x });

        // 5. Формируем 3D массив
        let output_3d = output_flat_relu.into_shape((self.output_size.0, self.output_size.1, self.num_filters)).unwrap();

        self.output_array = output_3d.clone();
        output_3d
    }

    pub fn backward(&mut self, error: Array3<f32>) -> Array3<f32> {
        let (out_w, out_h, num_filters) = self.output_size;
        let k = self.kernel_size;
        let in_c = self.input_size.2;
        let flat_size = k * k * in_c;

        // 1. Проходим через производную активации (Leaky ReLU)
        let mut error_relu = error.clone();
        for f in 0..num_filters {
            for y in 0..out_h {
                for x in 0..out_w {
                    if self.output_array[[x, y, f]] <= 0.0 {
                        error_relu[[x, y, f]] *= 0.01;
                    }
                }
            }
        }

        // 2. Вычисляем градиенты по весам (dL/dW)
        // Формула: dL/dW = Error * Input^T
        // Error shape: (num_filters, out_pixels)
        // Input_im2col shape: (flat_size, out_pixels)
        // Result: (num_filters, flat_size)
        
        let input_im2col = self.im2col_transpose(&self.input_array);
        let error_flat = error_relu.into_shape((num_filters, out_w * out_h)).unwrap();
        
        let grad_weights = error_flat.dot(&input_im2col.t()); // (num_filters, flat_size)
        let grad_weights_reshaped = grad_weights.into_shape((num_filters, k, k, in_c)).unwrap();

        self.kernel_changes -= &grad_weights_reshaped;

        // 3. Вычисляем градиенты по входу (dL/dInput)
        // Формула: dL/dInput = Kernels * Error
        // Kernels shape: (num_filters, flat_size)
        // Error shape: (num_filters, out_pixels)
        // Нужно: Kernels.T * Error -> (flat_size, out_pixels)
        
        let kernels_flat = self.kernels.view().into_shape((num_filters, flat_size)).unwrap();
        let grad_input_flat = kernels_flat.t().dot(&error_flat); // (flat_size, out_pixels)

        // 4. Собираем обратно в 3D
        self.col2im_transpose(&grad_input_flat)
    }

    pub fn zero(&mut self) {
        self.kernel_changes.fill(0.0);
    }

    pub fn update(&mut self, minibatch: usize) {
        self.kernel_changes /= minibatch as f32;
        self.kernels += &self.optimizer.weights_changes(&self.kernel_changes);
        self.kernel_changes = Array4::zeros((self.num_filters, self.kernel_size, self.kernel_size, self.input_size.2));
    }
}