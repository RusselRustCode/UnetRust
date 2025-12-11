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
use crate::utils::*;
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
        unimplemented!()
    }

    pub fn add_conv(&mut self, num_filters: usize, kernel_size: usize){
        todo!()
    }

    pub fn add_mxpl(&mut self, kernel_size: usize){
        todo!()
    }

    pub fn add_dense(&mut self, output_size: usize, activations: Activations, dropout: Option<f32>){
        todo!()
    }

    pub fn forward_prop(){
        todo!()
    }

    pub fn backward_prop(){
        todo!()
    }

    pub fn last_layer_error(){
        todo!()
    }

    pub fn update(){
        todo!()
    }

    pub fn output(){
        todo!()
    }

    pub fn get_accuracy(){
        todo!()
    }

    pub fn train(){
        todo!()
    }

    pub fn zero(){
        todo!()
    }

    pub fn save(){
        todo!()
    }

    pub fn top_n_accuracy(){
        todo!()
    }
}