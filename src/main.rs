mod activations;
mod utils;
use std::path::Path;

use utils::load_img;
use activations::{relu_deriv, sigmoid};
use ndarray::{array, Array1};
fn main() -> Result<(), String> {
    // let array: Array1<f32> = array![2.0, 3.0, -5.0];
    // let res = relu_deriv(array.clone());
    // let new_res = sigmoid(array);
    // println!("{}", res);

    // println!("{}", new_res);

    let image = load_img(Path::new("photo.png"))?;
    print!("{:?}", image.shape());
    Ok(())


}

