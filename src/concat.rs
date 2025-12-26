use ndarray::{Array3, Array4};
use serde::{Serialize, Deserialize};
use crate::layers::Layers;

#[derive(Serialize, Deserialize, Clone)]
pub struct Concatenate {
    // В слое конкатенации нет весов, но нам нужно знать,
    // какую часть входа берет левая ветка, а какую правая (для backward).
    // Однако в U-Net обычно склеивают выход энкодера и выход декодера.
    // Мы будем хранить размерность входа энкодера (skip connection),
    // чтобы знать, как резать градиент в backward.
    skip_shape: Option<(usize, usize, usize)>,
}

impl Concatenate {
    pub fn new() -> Self {
        Self {
            skip_shape: None,
        }
    }
    
    // Для U-Net нам часто нужно знать форму для следующего слоя
    pub fn get_output_shape(&self, decoder_shape: (usize, usize, usize)) -> (usize, usize, usize) {
        match self.skip_shape {
            Some(skip) => (decoder_shape.0, decoder_shape.1, decoder_shape.2 + skip.2),
            None => decoder_shape,
        }
    }
}

impl Concatenate {
    // forward принимает два массива: из декодера (upsampled) и из энкодера (skip)
    // Поскольку в нашем enum Layers слои принимают один вход, 
    // мы будем вызывать этот метод вручную внутри CNN, либо адаптировать Layers.
    // Но для чистоты, сделаем метод, принимающий два тензора.
    
    pub fn forward_pair(&mut self, decoder_input: Array3<f32>, encoder_input: Array3<f32>) -> Array3<f32> {
        // Сохраняем размерность энкодера для backward
        self.skip_shape = Some((encoder_input.dim().0, encoder_input.dim().1, encoder_input.dim().2));

        // Проверка на совпадение широты и долготы
        assert_eq!(decoder_input.shape()[0], encoder_input.shape()[0], "Height mismatch");
        assert_eq!(decoder_input.shape()[1], encoder_input.shape()[1], "Width mismatch");

        // Конкатенация по последней оси (каналам)
        // В ndarray для конкатенации 3D массивов проще всего использовать view и стекинг, 
        // или конвертацию в 2D, но так как нам нужен 3D выход, сделаем через map или stack
        // Самый эффективный способ в ndarray без лишних аллокаций - не самый тривиальный,
        // но надежный способ - создать новый массив и скопировать.
        
        let (h, w, c_dec) = decoder_input.dim();
        let c_enc = encoder_input.shape()[2];
        let c_out = c_dec + c_enc;

        let mut output = Array3::<f32>::zeros((h, w, c_out));
        
        // Копируем данные из декодера
        output.slice_mut(ndarray::s![.., .., 0..c_dec]).assign(&decoder_input);
        // Копируем данные из энкодера
        output.slice_mut(ndarray::s![.., .., c_dec..]).assign(&encoder_input);

        output
    }

    // В backward нам нужно вернуть два градиента: для декодера и для энкодера
    pub fn backward_pair(&self, grad_output: Array3<f32>) -> (Array3<f32>, Array3<f32>) {
        match self.skip_shape {
            Some(skip_shape) => {
                let c_dec = grad_output.shape()[2] - skip_shape.2;
                
                // Резаем градиент
                let grad_decoder = grad_output.slice(ndarray::s![.., .., 0..c_dec]).to_owned();
                let grad_encoder = grad_output.slice(ndarray::s![.., .., c_dec..]).to_owned();
                
                (grad_decoder, grad_encoder)
            },
            None => panic!("Concatenate backward called before forward"),
        }
    }
}