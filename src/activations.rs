use ndarray::{Array1};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activations {
    ReLU,
    Sigmoid,
    Softmax,
}

pub fn relu(x: Array1<f32>) -> Array1<f32>{
    return x.mapv(|xi | if xi >= 0.0 {xi} else {0.0});
}

pub fn relu_deriv(x: Array1<f32>) -> Array1<f32>{
    return x.mapv(|xi| if xi >= 0.0 {1.0} else {0.0});
}

pub fn sigmoid(x: Array1<f32>) -> Array1<f32>{
    return x.mapv(|xi| 1.0 / (1.0 + (-xi).exp()));
}

pub fn sigmoid_deriv(x: Array1<f32>) -> Array1<f32>{
    return x.mapv(|xi| xi * (1.0 - xi));
}

pub fn softmax(x: Array1<f32>) -> Array1<f32>{
    let max = x.fold(x[0], |acc, &xi| if xi > acc {xi} else {acc});
    let exps = x.mapv(|xi| (xi - max).exp());
    let sum = exps.sum();
    return exps/sum;
}

pub fn softmax_deriv(softmax_output: Array1<f32>, true_label: Array1<f32>) -> Array1<f32>{
    // return Array1::ones(x.len()); //Почему производная от softmax такая, пока что вопрос 
    // return softmax_output - true_label; //Производная при кросс-энтропии
    Array1::ones(softmax_output.len())
}


pub fn forward(x: Array1<f32>, activation: Activations) -> Array1<f32>{
    match activation{
        Activations::ReLU => relu(x),
        Activations::Sigmoid => sigmoid(x),
        Activations::Softmax => softmax(x),
        _=> panic!("Нет такой функции активации"),
    }
}

pub fn backward(x: Array1<f32>, activation: Activations, true_label: Option<Array1<f32>>) -> Array1<f32>{
    match activation{
        Activations::ReLU => relu_deriv(x),
        Activations::Sigmoid => sigmoid_deriv(x),
        Activations::Softmax => softmax_deriv(x, true_label.unwrap_or(Array1::<f32>::zeros(0))),
        _ => panic!("Нет такой функции активации и ее производной"),
    }
}