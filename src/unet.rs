use ndarray::Array3;
use crate::conv::ConvolutionLayer;
use crate::transposed_conv::TranposeConv; // Ваша реализация
use crate::mxpl::MaxPooling; // Ваш существующий слой
use crate::concat::Concatenate; // Слой конкатенации
use crate::optim::Optimizer;
use serde;
pub struct UNet {
    // --- Encoder ---
    pub enc_conv1: ConvolutionLayer,
    pub enc_pool1: MaxPooling,
    
    pub enc_conv2: ConvolutionLayer,
    pub enc_pool2: MaxPooling,
    
    pub enc_conv3: ConvolutionLayer,
    pub enc_pool3: MaxPooling,

    // --- Bottleneck ---
    pub bottleneck: ConvolutionLayer,

    // --- Decoder ---
    // Мы будем использовать TransposeConv для UpSampling.
    // Размерности рассчитаны для входа 256x256, padding=1 в свертках, stride=2 в TransposeConv
    
    pub dec_up3: TranposeConv,
    pub dec_conv3: ConvolutionLayer,
    
    pub dec_up2: TranposeConv,
    pub dec_conv2: ConvolutionLayer,
    
    pub dec_up1: TranposeConv,
    pub dec_conv1: ConvolutionLayer,

    // --- Final Layer ---
    // 1x1 свертка для приведения к числу классов (2 для Kvasir)
    pub final_layer: ConvolutionLayer,

    // --- Skip Connections ---
    pub concat3: Concatenate,
    pub concat2: Concatenate,
    pub concat1: Concatenate,

    // Сохраняем промежуточные выходы энкодера для Skip Connections
    pub skip1: Option<Array3<f32>>,
    pub skip2: Option<Array3<f32>>,
    pub skip3: Option<Array3<f32>>,
}

impl UNet {
    // Создает сеть под размер 256x256, 2 класса (фон, полип)
    pub fn new(optimizer: Optimizer) -> Self {
        let base_filters = 64;
        
        // --- Encoder ---
        // Вход: (256, 256, 1)
        // Conv 3x3, stride 1, padding 1 (чтобы размер остался 256)
        // Поскольку ваш ConvLayer в conv.rs может не считать padding, 
        // здесь предполагается, что ConvLayer::new принимает padding и корректно его обрабатывает.
        // Если нет, то output_size здесь нужно будет уменьшить (например, с 254).
        // Для расчета ниже я полагаю, что padding работает и размер сохраняется.
        
        let enc_conv1 = ConvolutionLayer::new((256, 256, 1), 3, 1, Some(1), base_filters, optimizer.clone());
        let enc_pool1 = MaxPooling::new((256, 256, base_filters), 2, 2);
        
        // (128, 128, 64) -> (128, 128, 128)
        let enc_conv2 = ConvolutionLayer::new((128, 128, base_filters), 3, 1, Some(1), base_filters * 2, optimizer.clone());
        let enc_pool2 = MaxPooling::new((128, 128, base_filters * 2), 2, 2);
        
        // (64, 64, 128) -> (64, 64, 256)
        let enc_conv3 = ConvolutionLayer::new((64, 64, base_filters * 2), 3, 1, Some(1), base_filters * 4, optimizer.clone());
        let enc_pool3 = MaxPooling::new((64, 64, base_filters * 4), 2, 2);

        // --- Bottleneck ---
        // (32, 32, 256) -> (32, 32, 512)
        let bottleneck = ConvolutionLayer::new((32, 32, base_filters * 4), 3, 1, Some(1), base_filters * 8, optimizer.clone());

        // --- Decoder ---
        
        // Up3: (32, 32, 512) -> (64, 64, 256)
        // ИСПРАВЛЕНИЕ: Добавлен kernel_size как кортеж (3, 3) вторым аргументом
        let dec_up3 = TranposeConv::new((32, 32, base_filters * 8), 3, 2, Some(1), base_filters * 4, optimizer.clone());
        
        // Conv3: Input (64, 64, 512) после конкатенации (256+256) -> (64, 64, 256)
        let dec_conv3 = ConvolutionLayer::new((64, 64, base_filters * 4 * 2), 3, 1, Some(1), base_filters * 4, optimizer.clone());
        
        // Up2: (64, 64, 256) -> (128, 128, 128)
        let dec_up2 = TranposeConv::new((64, 64, base_filters * 4), 3, 2, Some(1), base_filters * 2, optimizer.clone());
        let dec_conv2 = ConvolutionLayer::new((128, 128, base_filters * 2 * 2), 3, 1, Some(1), base_filters * 2, optimizer.clone());
        
        // Up1: (128, 128, 128) -> (256, 256, 64)
        let dec_up1 = TranposeConv::new((128, 128, base_filters * 2), 3, 2, Some(1), base_filters, optimizer.clone());
        let dec_conv1 = ConvolutionLayer::new((256, 256, base_filters * 2), 3, 1, Some(1), base_filters, optimizer.clone());

        // Final: (256, 256, 64) -> (256, 256, 2)
        let final_layer = ConvolutionLayer::new((256, 256, base_filters), 1, 1, Some(0), 2, optimizer.clone());

        UNet {
            enc_conv1, enc_pool1,
            enc_conv2, enc_pool2,
            enc_conv3, enc_pool3,
            bottleneck,
            dec_up3, dec_conv3,
            dec_up2, dec_conv2,
            dec_up1, dec_conv1,
            final_layer,
            concat3: Concatenate::new(),
            concat2: Concatenate::new(),
            concat1: Concatenate::new(),
            skip1: None,
            skip2: None,
            skip3: None,
        }
    }

    pub fn forward(&mut self, input: Array3<f32>) -> Array3<f32> {
        // --- Encoder ---
        let s1 = self.enc_conv1.forward(input);
        self.skip1 = Some(s1.clone()); // Сохраняем для Skip Connection
        let s1_pooled = self.enc_pool1.forward(s1);
        
        let s2 = self.enc_conv2.forward(s1_pooled);
        self.skip2 = Some(s2.clone());
        let s2_pooled = self.enc_pool2.forward(s2);
        
        let s3 = self.enc_conv3.forward(s2_pooled);
        self.skip3 = Some(s3.clone());
        let s3_pooled = self.enc_pool3.forward(s3);
        
        // --- Bottleneck ---
        let b = self.bottleneck.forward(s3_pooled);
        
        // --- Decoder ---
        // Block 3
        let d3_up = self.dec_up3.forward(b);
        // Skip connection (Concatenate)
        let s3_unwrapped = self.skip3.as_ref().unwrap().clone();
        let d3_cat = self.concat3.forward_pair(d3_up, s3_unwrapped);
        let d3 = self.dec_conv3.forward(d3_cat);
        
        // Block 2
        let d2_up = self.dec_up2.forward(d3);
        let s2_unwrapped = self.skip2.as_ref().unwrap().clone();
        let d2_cat = self.concat2.forward_pair(d2_up, s2_unwrapped);
        let d2 = self.dec_conv2.forward(d2_cat);
        
        // Block 1
        let d1_up = self.dec_up1.forward(d2);
        let s1_unwrapped = self.skip1.as_ref().unwrap().clone();
        let d1_cat = self.concat1.forward_pair(d1_up, s1_unwrapped);
        let d1 = self.dec_conv1.forward(d1_cat);
        
        // --- Output ---
        let mut out = self.final_layer.forward(d1);
        
        // Применяем Softmax по последней оси (каналам классов)
        out = self.softmax_3d(out);
        
        out
    }

    pub fn backward(&mut self, grad_loss: Array3<f32>) {
        // Обратный проход от конца к началу
        
        // 1. Final Layer
        let mut grad = self.final_layer.backward(grad_loss);
        
        // 2. Decoder Block 1
        grad = self.dec_conv1.backward(grad);
        let (g_up1, g_skip1) = self.concat1.backward_pair(grad);
        grad = self.dec_up1.backward(g_up1);
        
        // 3. Decoder Block 2
        grad = self.dec_conv2.backward(grad);
        let (g_up2, g_skip2) = self.concat2.backward_pair(grad);
        grad = self.dec_up2.backward(g_up2);
        
        // 4. Decoder Block 3
        grad = self.dec_conv3.backward(grad);
        let (g_up3, g_skip3) = self.concat3.backward_pair(grad);
        grad = self.dec_up3.backward(g_up3);
        
        // 5. Bottleneck
        grad = self.bottleneck.backward(grad);
        
        // 6. Encoder Block 3 (пропускаем через MaxPool backward)
        grad = self.enc_pool3.backward(grad);
        grad = self.enc_conv3.backward(grad);
        // Суммируем градиент с Skip Connection
        grad += &g_skip3;

        // 7. Encoder Block 2
        grad = self.enc_pool2.backward(grad);
        grad = self.enc_conv2.backward(grad);
        grad += &g_skip2;

        // 8. Encoder Block 1
        grad = self.enc_pool1.backward(grad);
        grad = self.enc_conv1.backward(grad);
        grad += &g_skip1;
    }

    // Обновление весов для всех слоев
    pub fn update(&mut self, batch_size: usize) {
        self.enc_conv1.update(batch_size);
        self.enc_conv2.update(batch_size);
        self.enc_conv3.update(batch_size);
        self.bottleneck.update(batch_size);
        
        self.dec_up3.update(batch_size);
        self.dec_conv3.update(batch_size);
        self.dec_up2.update(batch_size);
        self.dec_conv2.update(batch_size);
        self.dec_up1.update(batch_size);
        self.dec_conv1.update(batch_size);
        
        self.final_layer.update(batch_size);
    }
    
    // Обнуление градиентов
    pub fn zero(&mut self) {
        self.enc_conv1.zero();
        self.enc_conv2.zero();
        self.enc_conv3.zero();
        self.bottleneck.zero();
        
        self.dec_up3.zero();
        self.dec_conv3.zero();
        self.dec_up2.zero();
        self.dec_conv2.zero();
        self.dec_up1.zero();
        self.dec_conv1.zero();
        
        self.final_layer.zero();
        
        self.enc_pool1.zeros();
        self.enc_pool2.zeros();
        self.enc_pool3.zeros();
    }

    // Вспомогательная функция для Softmax по каналам
    fn softmax_3d(&self, input: Array3<f32>) -> Array3<f32> {
        let (h, w, c) = input.dim();
        let mut output = input.clone();
        
        for y in 0..h {
            for x in 0..w {
                // Берем срез по каналам для пикселя (x,y)
                let mut slice = output.slice_mut(ndarray::s![x, y, ..]);
                
                // Находим максимум для стабильности
                let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                
                // Вычисляем экспоненты
                let exps: Vec<f32> = slice.iter().map(|v| (v - max_val).exp()).collect();
                let sum: f32 = exps.iter().sum();
                
                // Нормализуем
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = exps[i] / sum;
                }
            }
        }
        output
    }
}