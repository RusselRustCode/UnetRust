use serde::{Deserialize, Serialize};

use crate::mxpl::{MaxPooling};
use crate::conv::{ConvolutionLayer};
use crate::dense::{DenseLayer};
use::std::fmt::{Debug, Formatter};
use crate::transposed_conv::TranposeConv;

#[derive(Serialize, Deserialize)]
pub enum Layers {
    Mxpl(MaxPooling),
    Dense(DenseLayer),
    Conv(ConvolutionLayer),
    TransposeConv(TranposeConv),
}

impl Debug for Layers{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Layers::Mxpl(layer) => write!(f, "{:?}", layer),
            Layers::Conv(layer) => write!(f, "{:?}", layer),
            Layers::Dense(layer) => write!(f, "{:?}", layer),
            Layers::TransposeConv(layer) => write!(f, "{:?}", layer),
        }
    }
}




