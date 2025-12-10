use std::fmt::{Debug, Formatter};
use serde::{Deserialize, Serialize};
use crate::{optim::{Optim2D, Optimizer}, utils::outer};
use ndarray::{Array1, Array2};
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};
use crate::activations::{Activations, forward, backward};

#[derive(Serialize, Deserialize)]
pub struct DenseLayer{
    input_size: usize,
    pub output_size: usize,
    #[serde(skip)]
    input: Array1<f32>,
    #[serde(skip)]
    pub output: Array1<f32>,
    biases: Array1<f32>,
    weights: Array2<f32>,
    #[serde(skip)]
    bias_changes: Array1<f32>,
    #[serde(skip)]
    weights_changes: Array2<f32>,
    activation: Activations,
    pub trans_shape: (usize, usize, usize),
    optimizer: Optim2D,
    dropout: Option<f32>,
    #[serde(skip)]
    dropout_mask: Array1<f32>,
}

impl Debug for DenseLayer{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        s.push_str(&format!("Dense Layer\n"));
        s.push_str(&format!("input size = {}\n", self.input_size));
        s.push_str(&format!("output size = {}\n", self.output_size));
        s.push_str(&format!("activation is {:?}\n", self.activation));
        match self.dropout {
            Some(dropout) => s.push_str(&format!("Dropout = {}", dropout)),
            None => s.push_str(&format!("Нет dropout")),
        }        
        return write!(f, "{}", s);
    }
}

impl DenseLayer{
    pub fn new(input_size: usize, output_size: usize, activation: Activations, optim_alg: Optimizer, dropout: Option<f32>, trans_shape: (usize, usize, usize)) -> Self{
        let thread_rng = &mut rand::thread_rng();

        //He/Kaiming
        let normal = Normal::new(0.0, (2.0 / input_size as f32).sqrt()).unwrap();
        let weights = Array2::<f32>::from_shape_fn((output_size, input_size),|_| normal.sample(thread_rng));

        let biases = Array1::<f32>::from_elem(output_size, 0.01);


        let optimizer = Optim2D::new(optim_alg, input_size, output_size);

        return Self{
            input_size: input_size,
            output_size: output_size,
            input: Array1::<f32>::zeros(input_size),
            output: Array1::<f32>::zeros(output_size),
            biases: biases,
            weights: weights,
            bias_changes: Array1::<f32>::zeros(output_size),
            weights_changes: Array2::<f32>::zeros((output_size, input_size)),
            activation: activation,
            trans_shape: trans_shape,
            optimizer: optimizer,
            dropout: dropout,
            dropout_mask: Array1::<f32>::zeros(output_size),   
        };
    }

    pub fn forward(&mut self, input: Array1<f32>) -> Array1<f32>{
        if self.dropout.is_some(){
            let dropout = self.dropout.unwrap();
            let mut rng = thread_rng();
            self.dropout_mask = Array1::<f32>::from_shape_fn((self.output_size), |_| rng.r#gen::<f32>());
            self.dropout_mask = self.dropout_mask.mapv(|x| if x < dropout {0.0} else {1.0});
            let logits = self.weights.dot(&input) + &self.biases;
            self.output = forward(logits, self.activation);
            self.output = &self.output * &self.dropout_mask;
            self.input = input;
            return self.output.clone();
        }
        else {
            let logits = self.weights.dot(&input) + &self.biases;
            self.output = forward(logits, self.activation);
            self.input = input;
            return self.output.clone();
        }
    }

    pub fn backpropagate(&mut self, error: Array1<f32>, training: bool, true_label: Option<Array1<f32>>) -> Array1<f32>{
        let mut error = error;
        if self.dropout.is_some() && training{
            error *= &self.dropout_mask;
        }
        error *= &backward(self.output.clone(), self.activation, None);
        let prev_error = self.weights.t().dot(&error);
        self.weights_changes -= &(outer(error.clone(), self.input.clone()));
        self.bias_changes -= &error;

        return prev_error;

    }

    pub fn zero(&mut self){
        self.bias_changes = Array1::<f32>::zeros(self.output_size);
        self.weights_changes = Array2::<f32>::zeros((self.output_size, self.input_size));
        self.output = Array1::<f32>::zeros(self.output_size);
    }

    pub fn update(&mut self, minibatchsize: usize, lr: f32){
        self.weights_changes /= minibatchsize as f32;
        self.bias_changes /= minibatchsize as f32;
        self.weights_changes += &self.optimizer.weights_changes(&self.weights_changes);
        self.bias_changes += &self.optimizer.bias_changes(lr, &self.bias_changes);
        self.weights_changes = Array2::<f32>::zeros((self.output_size, self.input_size));
        self.bias_changes = Array1::<f32>::zeros(self.output_size);
    }
}