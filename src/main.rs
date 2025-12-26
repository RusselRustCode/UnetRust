use cnn_rust::cnn::*;
use cnn_rust::mnist::load_mnist;
use  cnn_rust::optim::*;
use cnn_rust::activations::*;


fn main() -> Result<(), String> {
    let data = load_mnist("./data/");

    let hyperparameters = HyperParams {
        batch_size:32,
        epochs: 4,
        optimizer: Optimizer::Adam(0.0001, 0.9, 0.999, Some(1e-8)),
        ..HyperParams::default()
    };


    let mut cnn = CNN::new(data, hyperparameters);
    cnn.set_input_shape(vec![28, 28, 1]);
    cnn.add_conv(32, 3);
    cnn.add_mxpl(2);
    cnn.add_dense(128, Activations::LeakyReLU, None);
    cnn.add_dense(64, Activations::LeakyReLU, None);
    cnn.add_dense(10, Activations::Softmax, None);

    cnn.train_mnist();

    Ok(())

    //добавить LeakyReLU, поиграться с Dropout, поработать с cudarc, допилить padding для conv.rs

}

