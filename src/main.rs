use cnn_rust::cnn::*;
use cnn_rust::mnist::load_mnist;
use  cnn_rust::optim::*;
use cnn_rust::activations::*;


fn main() -> Result<(), String> {
    let data = load_mnist("./data/");

    let hyperparameters = HyperParams {
        batch_size:32,
        epochs: 10,
        optimizer: Optimizer::Adam(5e-4, 0.9, 0.999, Some(1e-8)),
        ..HyperParams::default()
    };


    let mut cnn = CNN::new(data, hyperparameters);
    cnn.set_input_shape(vec![28, 28, 1]);
    cnn.add_conv(32, 3);
    cnn.add_mxpl(2);
    cnn.add_dense(128, Activations::ReLU, Some(0.25));
    cnn.add_dense(64, Activations::ReLU, Some(0.25));
    cnn.add_dense(10, Activations::Softmax, None);

    cnn.train_mnist();

    Ok(())

}

