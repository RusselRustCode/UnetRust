use core::panic;
use std::io::Write;
use std::fs::File;
use std::fmt::{Formatter, Debug};
use std::time::{SystemTime, UNIX_EPOCH};
use std::default::{self, Default};
use std::vec;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array3};
use serde::{Deserialize, Serialize};
use crate::activations::{self, Activations};
use crate::{conv, dense, layers, mxpl, utils::*};
use crate::optim::Optimizer;
use crate::mnist::*;

use crate::{
    conv::ConvolutionLayer, dense::DenseLayer, layers::Layers,
    mxpl::MaxPooling
};

pub struct HyperParams{
    pub batch_size: usize,
    pub epochs: usize,
    pub optimizer: Optimizer,
    pub saving_strategy: SavingStrategy,
    pub name: String,
    pub verbose: bool,
}

impl Default for HyperParams{
    fn default() -> Self {
        HyperParams { 
            batch_size: 32,
            epochs: 10,
            optimizer: Optimizer::Adam(0.9, 0.999, 1e-8, None),
            saving_strategy: SavingStrategy::Never,
            name: String::from("model"),
            verbose: true,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct CNN {
    layers: Vec<Layers>,
    layer_order: Vec<String>,
    data: TrainingData,
    minibatch_size: usize,
    creation_time: SystemTime,
    saving_strategy: SavingStrategy,
    training_history: Vec<f32>,
    testing_history: Vec<f32>,
    time_history: Vec<usize>,
    name: String,
    verbose: bool,
    optimizer: Optimizer,
    epochs: usize,
    input_shape: (usize, usize, usize),
}

impl Debug for CNN{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        let time = self.creation_time.duration_since(UNIX_EPOCH).unwrap().as_millis();
        s.push_str(&format!("File: models/{}_{}.json\n", self.name, time));
        s.push_str(&format!("Время: {}\n", time));
        s.push_str(&format!("Размер батча: {}\n", self.minibatch_size));
        s.push_str(&format!("Обучающая выборка: {}\n", self.data.trn_size));
        s.push_str(&format!("Тестовая выборка: {}\n", self.data.test_size));
        s.push_str(&format!("\nСлои:\n"));
        
        for layer in &self.layers {
            s.push_str(&format!("{:?}\n", layer));
        }

        s.push_str(&format!("Точность на обучающей выборке: {:?}\n", self.training_history));
        s.push_str(&format!("Точность на тестовой выборке: {:?}\n", self.testing_history));
        s.push_str(&format!("Время потраченное: {:?}\n", self.time_history));

        write!(f, "{}", s)
    }
}

impl CNN{
    pub fn new(data: TrainingData, params: HyperParams) -> Self{
        let creation_time = SystemTime::now();
        
        return Self { 
            layers: vec![],
            layer_order: vec![],
            data: data,
            minibatch_size: params.batch_size,
            creation_time: creation_time,
            saving_strategy: params.saving_strategy,
            training_history: vec![],
            testing_history: vec![],
            time_history: vec![],
            name: params.name,
            verbose: params.verbose,
            optimizer: params.optimizer,
            epochs: params.epochs,
            input_shape: (0, 0, 0)
         };
    }


    pub fn load(model_file_name: &str) -> Self{
        let model_file = File::open(model_file_name).unwrap();
        let cnn: CNN = serde_json::from_reader(model_file).unwrap();
        return cnn;
    }

    pub fn set_input_shape(&mut self, input_shape: Vec<usize>){
        let mut iter = input_shape.into_iter();
        self.input_shape = (
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
        );

    }

    pub fn add_conv(&mut self, num_filters: usize, kernel_size: usize){
        if self.input_shape.0 == 0{
            panic!("Размер для входа не задан! Вызовите set_input_shape");
        }

        let input_size: (usize, usize, usize) = match self.layers.last(){
            Some(Layers::Conv(conv_layer)) => conv_layer.output_size,
            Some(Layers::Mxpl(maxpooling_layer)) => maxpooling_layer.output_size,
            Some(Layers::Dense(dense_layer)) => panic!("Не может Сверточный слой следовать за Полносвязным!"),
            Some(Layers::TransposeConv(_)) => unimplemented!(),
            None => self.input_shape,
        };
        
        let conv_layer = ConvolutionLayer::new(input_size, kernel_size, 1, None, num_filters, self.optimizer.clone());
        self.layers.push(Layers::Conv(conv_layer));
        self.layer_order.push(String::from("convolutional"));
    }

    pub fn add_mxpl(&mut self, kernel_size: usize){
        if self.input_shape.0 == 0{
            panic!("Размер для входа не задан! Вызовите set_input_shape");
        }

        let input_size: (usize, usize, usize) = match self.layers.last(){
            Some(Layers::Conv(conv_layer)) => conv_layer.output_size,
            Some(Layers::Mxpl(maxpooling_layer)) => maxpooling_layer.output_size,
            Some(Layers::Dense(_)) => panic!("Max pooling слой не может идти после полносвязного слоя"),
            Some(Layers::TransposeConv(trans_conv)) => panic!("Max Pooling слоя не может идти после Transposed Convolutional слоя"),
            None => self.input_shape,
        };
        

        let mxpl_layer: MaxPooling = MaxPooling::new(input_size, kernel_size, 1);
        self.layers.push(Layers::Mxpl(mxpl_layer));
        self.layer_order.push(String::from("maxpooling"));
        
    }

    pub fn add_dense(&mut self, output_size: usize, activation: Activations, dropout: Option<f32>){
        if self.input_shape.0 == 0{
            panic!("");
        }

        let trans_shape: (usize, usize, usize) = match self.layers.last(){
            Some(Layers::Conv(conv_layer)) => conv_layer.output_size,
            Some(Layers::Mxpl(maxpooling_layer)) => maxpooling_layer.output_size,
            Some(Layers::Dense(dense_layer)) => (dense_layer.output_size, 1, 1),
            Some(Layers::TransposeConv(trans_layer)) => trans_layer.output_size,
            None => self.input_shape
        };
        let flatten_data = trans_shape.0 * trans_shape.1 * trans_shape.2;
        let fcl: DenseLayer = DenseLayer::new(flatten_data, output_size, activation, self.optimizer, dropout, trans_shape);
        self.layers.push(Layers::Dense(fcl));
        self.layer_order.push(String::from("dense"));
    }


    pub fn add_trans_conv(&mut self){
        todo!()
    }

    pub fn forward_prop(&mut self, image: Array3<f32>, training: bool) -> Array1<f32>{
        let mut output = image;
        let mut flat_data: Array1<f32> = output.clone().into_shape((output.len())).unwrap();
        for layer in &mut self.layers{
            match layer {
                Layers::Conv(conv_layer) => {
                    output = conv_layer.forward(output);
                    flat_data = output.clone().into_shape(output.len()).unwrap();
                },
                Layers::Mxpl(maxpooling_layer) => {
                    output = maxpooling_layer.forward(output);
                    flat_data = output.clone().into_shape(output.len()).unwrap();
                },
                Layers::Dense(dense_layer) => flat_data = dense_layer.forward(flat_data),
                Layers::TransposeConv(trans_conv_layer) => unimplemented!(),
            }
        }
        return flat_data;
    }

    
    pub fn last_layer_error(&mut self, label:usize) -> Array1<f32>{
        let size: usize = match self.layers.last().unwrap() {
            Layers::Dense(dense_layer) => dense_layer.output_size,
            _ => panic!("Последний слой полносвзяный должен быть!"),
        };
        let desired = Array1::<f32>::from_shape_fn(size, |i| (label == i) as usize as f32);
        return self.output() - desired;
    }

    pub fn backward_prop(&mut self, label:usize, training: bool){
        let mut flat_error = self.last_layer_error(label);
        let mut error = flat_error.clone().into_shape((1, 1, flat_error.len())).unwrap();
        let size = match self.layers.last().unwrap() {
            Layers::Dense(dl) => dl.output_size,
            _ => panic!(),
        };
        let true_label = Array1::from_shape_fn(size, |i| if i == label {1.0} else {0.0});
        for layer in self.layers.iter_mut().rev(){
            match layer {
                Layers::Conv(conv_layer) => {
                    error = conv_layer.backward(error);
                },
                Layers::Mxpl(maxpooling_layer) => {
                    error = maxpooling_layer.backward(error);
                },
                Layers::Dense(dense_layer) => {
                    flat_error = dense_layer.backpropagate(flat_error, training, Some(true_label.clone()));
                    error = flat_error.clone().into_shape(dense_layer.trans_shape).unwrap();
                },
                Layers::TransposeConv(trans_layer) => unimplemented!(),
            }
        }
        
    }


    pub fn update(&mut self, minibatch: usize){
        for layer in &mut self.layers{
            match layer{
                Layers::Conv(conv_layer) => conv_layer.update(minibatch),
                Layers::Mxpl(_) => {},
                Layers::Dense(dense_layer) => dense_layer.update(minibatch, 0.001),
                Layers::TransposeConv(trans_conv) => unimplemented!(),
            }
        }
    }

    pub fn output(&self) -> Array1<f32>{
        match self.layers.last().unwrap(){
            Layers::Conv(_) => panic!("Не может последний слой быть ConvLayer"),
            Layers::Mxpl(_) => panic!("Не может последний слой быть MaxPooling"),
            Layers::TransposeConv(_) => panic!("Не может последний слой быть TransposedConvLayer"),
            Layers::Dense(dense_layer) => dense_layer.output.clone(),

        }
    }

    pub fn get_accuracy(&self, label:usize) -> f32{
        let mut max = 0.0f32;
        let mut max_indx = 0usize;
        let output = self.output();
        for i in 0..output.len(){
            if output[[i]] > max{
                max = output[[i]];
                max_indx = i;
            }
        }
        (max_indx == label) as usize as f32
    
    }

    pub fn train_mnist(&mut self, ){
        let mut best_train_accuracy = self.training_history.last().unwrap_or(&0.0).clone();
        let mut best_test_accuracy = self.testing_history.last().unwrap_or(&0.0).clone();

        for epoch in 0..self.epochs{
            self.zero();
            let pb = ProgressBar::new((self.data.trn_size / self.minibatch_size) as u64);
            if self.verbose{
                pb.set_style(ProgressStyle::default_bar()
                .template(&format!("Epoch {}: [{{bar:.cyan/blue}}] {{pos}}/{{len}} - ETA: {{eta}} - acc: {{msg}}", epoch))
                .unwrap()
                .progress_chars("#>-"))
            }
            let mut avg_accuracy = 0.0;
            for i in 0..self.data.trn_size{
                let (image, label) = get_rand_train_image(&self.data);
                let label = *self.data.classes.get(&label).unwrap();
                self.forward_prop(image, true);
                self.backward_prop(label, true);


                avg_accuracy+= self.get_accuracy(label);


                if i % self.minibatch_size == self.minibatch_size - 1{
                    self.update(self.minibatch_size);

                    if self.verbose{
                        pb.inc(1);
                        pb.set_message(format!("{:.1}%", avg_accuracy / (i + 1) as f32 * 100.0));
                    }
                }
                match self.saving_strategy{
                    SavingStrategy::EveryNthEpoch(full_save, n) => {
                        let every_n = (self.data.trn_size as f32 * n) as usize;
                        if i % every_n == every_n - 1{
                            self.save(full_save);
                        }
                    },
                    _ => {}
                }

            }
            avg_accuracy /= self.data.trn_size as f32;
            if self.verbose {
                pb.set_message(format!("{:.1}% - Testing...", avg_accuracy));
            }

            //Test
            let mut avg_test_acc = 0.0;
            for _i in 0..self.data.test_size {
                let (image, label) = get_rand_test_image(&self.data);
                let label = *self.data.classes.get(&label).unwrap();
                self.forward_prop(image, false);
                
                avg_test_acc += self.get_accuracy(label);
            }
            avg_test_acc /= self.data.test_size as f32;
            if self.verbose {
                pb.finish_with_message(format!("{:.1}% - Test: {:.1}%", avg_accuracy * 100.0, avg_test_acc * 100.0));
            }

            self.training_history.push(avg_accuracy);
            self.testing_history.push(avg_test_acc);
            let duration = SystemTime::now().duration_since(self.creation_time).unwrap();
            self.time_history.push(duration.as_secs() as usize);
            match self.saving_strategy {
                SavingStrategy::EveryEpoch(full_save) => {
                    self.save(full_save);
                }
                SavingStrategy::BestTrainingAccuracy(full_save) => {
                    if avg_accuracy > best_train_accuracy {
                        best_train_accuracy = avg_accuracy;
                        self.save(full_save);
                    } else {
                        self.save(false);
                    }
                }
                SavingStrategy::BestTestingAccuracy(full_save) => {
                    if avg_test_acc > best_train_accuracy {
                        best_train_accuracy = avg_test_acc;
                        self.save(full_save);
                    } else {
                        self.save(false);
                    }
                }
                _ => {}
            }

        }
    }

    pub fn zero(&mut self){
        for layer in &mut self.layers{
            match layer{
                Layers::Conv(conv_layer) => conv_layer.zero(),
                Layers::Mxpl(maxpooling_layer) => maxpooling_layer.zeros(),
                Layers::Dense(dense_layer) => dense_layer.zero(),
                Layers::TransposeConv(trans_layer) => unimplemented!(),
            }
        }
    }

    pub fn save(&self, full_save: bool){
        std::fs::create_dir_all("models").unwrap();
        let time = self.creation_time.duration_since(UNIX_EPOCH).unwrap().as_millis();  
        if full_save{
            let model_file_name = format!("models/{}_{}.txt", self.name, time);
            let mut metadata_file = std::fs::File::create(&model_file_name).unwrap();
            write!(metadata_file, "{:?}", self).unwrap();
        }
    }
}