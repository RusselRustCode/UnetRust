use ndarray::{Array1, Array2, Array4};
use serde::{Serialize, Deserialize};
use std::{fmt::{Debug, Formatter}, usize};
//Допилить Adagrad и AdaDelta
#[derive(Clone, Copy, Serialize, Deserialize)] 
pub enum Optimizer{
    SGD(f32),
    Momentum(f32, f32),
    RMSProp(f32, f32, Option<f32>),
    Adam(f32, f32, f32, Option<f32>),
    AdaGrad(f32, f32),
    AdaDelta(f32, f32)
}


impl Debug for Optimizer{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        match self{
            Optimizer::SGD(lr) => {
                s.push_str(&format!("SGD\n"));
                s.push_str(&format!("Learning Rate: {}\n", lr));
            },
            Optimizer::Momentum(lr, mu) => {
                s.push_str(&format!("Momentum\n"));
                s.push_str(&format!("Learning Rate {}\n", lr));
                s.push_str(&format!("Momentun: {}\n", mu));
            },
            Optimizer::RMSProp(lr, rho, eps) => {
                s.push_str(&format!("RMSProp\n"));
                s.push_str(&format!("Learning Rate: {}\n", lr));
                s.push_str(&format!("Rho: {}\n", rho));
                s.push_str(&format!("Epsilon: {}\n", eps.unwrap_or(1e-8)));
            },
            Optimizer::Adam(lr, beta1, beta2, eps) => {
                s.push_str(&format!("Adam\n"));
                s.push_str(&format!("Learning  Rate: {}\n", lr));
                s.push_str(&format!("Beta 1: {}\n", beta1));
                s.push_str(&format!("Beta 2: {}\n", beta2));
                s.push_str(&format!("Epsilon: {}\n", eps.unwrap_or(1e-8)));
            },
            _ =>{

            }
        }
        write!(f, "{}", s)

    }
}

//Чтобы не забыть почему 2d и 4d!
//Дело в том что входные данные для conv2d(двумерной свертки) должны иметь 4-мерную структуру, которая выглядит след образом:
//[размер батча, высота, ширина, каналы] -> [32, 28, 28, 3] -> Означает, что мы используем мини батч размером 32 изорбражения, где каждое изображение 
//имеет размер 28 на 28 пикселей и каждая картинка является RGB изображением значит имеет 3 цветных канала(красный, зеленый, синий). 

#[derive(Serialize, Deserialize)]
pub struct Optim2D{
    pub alg: Optimizer,
    pub history_grad1: Array2<f32>,
    pub history_grad2: Array2<f32>,
    pub t: i32,
    pub beta1: bool,
    pub beta2: bool,
}

#[derive(Serialize, Deserialize)]
pub struct Optim4D{
    pub alg: Optimizer,
    pub history_grad1: Array4<f32>,
    pub history_grad2: Array4<f32>,
    pub t: i32,
    pub beta1: bool,
    pub beta2: bool,
}


impl Optim2D{
    pub fn new(alg: Optimizer, input_size: usize, output_size: usize, weights: &Array2<f32>) -> Self{
        let shape = weights.raw_dim();
        return Self{
            alg: alg,
            history_grad1: Array2::zeros(shape),
            history_grad2: Array2::zeros(shape),
            t: 0,
            beta1: false,
            beta2: false,
        };
    }


    pub fn weights_changes(&mut self, gradients: &Array2<f32>) -> Array2<f32>{
        match self.alg{
            Optimizer::SGD(lr) => {
                return gradients * lr;
            },
            Optimizer::Momentum(lr, momentum) => {
                self.history_grad1 = &self.history_grad1 * momentum + lr * gradients;
                return &self.history_grad1 * lr; // ut = momentum * ut-1 + lr * grad
            },
            Optimizer::RMSProp(lr, rho, eps) => {
                self.history_grad1 = &self.history_grad1 * rho;
                self.history_grad1 += &(gradients.mapv(|xi| xi.powf(2.0)) * (1.0 - rho));
                return lr * gradients / (self.history_grad1.mapv(|xi| xi.sqrt() + eps.unwrap_or(1e-8)));
                // ut = -lr * grad / sqrt(Gt-1 + eps), Gt = rho * Gt-1 + (1- rho) * grad^2
            },
            Optimizer::Adam(lr, beta1, beta2, eps) =>{
                assert_eq!(
                    self.history_grad1.shape(),
                    gradients.shape(),
                    "history_grad1 and gradients have incompatible shapes, ОШИБКА"
                );
                self.t += 1;
                // self.history_grad1 = beta1 * &self.history_grad1 + &(gradients.mapv(|xi| xi * (1.0 - beta1)));
                self.history_grad1 = beta1 * &self.history_grad1 + (1.0 - beta1) * gradients;
                // self.history_grad2 = beta2 * &self.history_grad2 + &(gradients.mapv(|xi| xi.powf(2.0) * (1.0 - beta2)));
                self.history_grad2 = beta2 * &self.history_grad2 + (1.0 - beta2) * gradients.mapv(|xi| xi.powf(2.0));

                //Скорректированные оценки смещения
                //Проблема в том что на начальных этапах, особенно на t = 1, 2, .. m(history_grad1), v(history_grad2) инициализируются 0, поэтому оценки mt и vt
                //сильно смещены к нулю(занижены) и без коррекции это может привести к слишком большим или нестабильным изменениям
                //например, при t = 1, beta = 0.9: beta * m0 + (1 - beta) * grad => 0.9 * 0 + 0.1 * grad = 0.1 * grad(всего лишь 10 процентов от градиента, очень мало)
                //скорректированные оценки: mt_hat = mt / (1 - beta1^t), vt_hat = vt / (1 - beta2^)
                let beta1_done = if self.beta1{
                    0.0
                }
                else{
                    let pow = beta1.powi(self.t);
                    if pow < 0.001{
                        self.beta1 = true;
                    }
                    pow
                };

                let beta2_done = if self.beta2{
                    0.0
                }
                else{
                    let pow = beta2.powi(self.t);
                    if pow < 0.001{
                        self.beta2 = true;
                    }
                    pow
                };
                
                let corrected_velocity = &self.history_grad1 / (1.0 - beta1_done);
                let corrected_velocity2 = &self.history_grad2 / (1.0 - beta2_done);

                return lr * &corrected_velocity / (corrected_velocity2.mapv(|xi| xi.sqrt()) + eps.unwrap_or(1e-8));
            },
            _=> {
                unimplemented!()
            }
        }
    }

    pub fn bias_changes(&mut self, lr: f32, gradients: &Array1<f32>) -> Array1<f32>{
        match self.alg{
            Optimizer::SGD(lr) => {
                return lr * gradients;
            },
            Optimizer::Momentum(lr, _) => {
                return lr * gradients;
            },
            Optimizer::Adam(lr, _, _, _) => {
                return lr * gradients;
            },
            Optimizer::RMSProp(lr, _, _) => {
                return lr * gradients;
            },
            _ => {
                unimplemented!();
            }
        }
    }
}

impl Optim4D{
    pub fn new(alg: Optimizer, size: (usize, usize, usize, usize)) -> Self{
        return Self{
            alg: alg, 
            history_grad1: Array4::zeros((size.0, size.1, size.2, size.3)),
            history_grad2: Array4::zeros((size.0, size.1, size.2, size.3)),
            t: 0,
            beta1: false, 
            beta2: false,
        };
    }

    
    pub fn weights_changes(&mut self, gradients: &Array4<f32>) -> Array4<f32>{
        match self.alg{
            Optimizer::SGD(lr) => {
                return gradients * lr;
            },
            Optimizer::Momentum(lr, momentum) => {
                self.history_grad1 = &self.history_grad1 * momentum + lr * gradients;
                return &self.history_grad1 * lr; // ut = momentum * ut-1 + lr * grad
            },
            Optimizer::RMSProp(lr, rho, eps) => {
                self.history_grad1 = &self.history_grad1 * rho;
                self.history_grad1 += &(gradients.mapv(|xi| xi.powf(2.0)) * (1.0 - rho));
                return lr * gradients / (self.history_grad1.mapv(|xi| xi.sqrt() + eps.unwrap_or(1e-8)));
                // ut = -lr * grad / sqrt(Gt-1 + eps), Gt = rho * Gt-1 + (1- rho) * grad^2
            },
            Optimizer::Adam(lr, beta1, beta2, eps) =>{
                self.t += 1;
                self.history_grad1 = beta1 * &self.history_grad1 + &(gradients.mapv(|xi| xi * (1.0 - beta1)));
                self.history_grad2 = beta2 * &self.history_grad2 + &(gradients.mapv(|xi| xi.powf(2.0) * (1.0 - beta2)));

                //Скорректированные оценки смещения
                //Проблема в том что на начальных этапах, особенно на t = 1, 2, .. m(history_grad1), v(history_grad2) инициализируются 0, поэтому оценки mt и vt
                //сильно смещены к нулю(занижены) и без коррекции это может привести к слишком большим или нестабильным изменениям
                //например, при t = 1, beta = 0.9: beta * m0 + (1 - beta) * grad => 0.9 * 0 + 0.1 * grad = 0.1 * grad(всего лишь 10 процентов от градиента, очень мало)
                //скорректированные оценки: mt_hat = mt / (1 - beta1^t), vt_hat = vt / (1 - beta2^)
                let beta1_done = if self.beta1{
                    0.0
                }
                else{
                    let pow = beta1.powi(self.t);
                    if pow < 0.001{
                        self.beta1 = true;
                    }
                    pow
                };

                let beta2_done = if self.beta2{
                    0.0
                }
                else{
                    let pow = beta2.powi(self.t);
                    if pow < 0.001{
                        self.beta2 = true;
                    }
                    pow
                };
                
                let corrected_velocity = &self.history_grad1 / (1.0 - beta1_done);
                let corrected_velocity2 = &self.history_grad2 / (1.0 - beta2_done);

                return lr * &corrected_velocity / (corrected_velocity2.mapv(|xi| xi.sqrt()) + eps.unwrap_or(1e-8));
            },
            _=> {
                unimplemented!()
            }
        }
    }


    pub fn bias_changes(&mut self, lr: f32, gradients: Array1<f32>) -> Array1<f32>{
        match self.alg{
            Optimizer::SGD(lr) => {
                return lr * gradients;
            },
            Optimizer::Momentum(lr, _) => {
                return lr * gradients;
            },
            Optimizer::Adam(lr, _, _, _) => {
                return lr * gradients;
            },
            Optimizer::RMSProp(lr, _, _) => {
                return lr * gradients;
            },
            _ => {
                unimplemented!();
            }
        }
    }
}






