use aha::utils::interpolate::interpolate_nearest_1d;
use anyhow::Result;
use candle_core::Tensor;

#[test]
fn interpolate_test() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test interpolate interpolate_test -r -- --nocapture
    let device = &candle_core::Device::Cpu;
    let input = Tensor::arange(0.0f32, 10.0f32, device)?;
    let input_1d = input.reshape((1, 1, 10))?;
    println!("input_1d: {}", input_1d);
    let x_nearest_1d = interpolate_nearest_1d(&input_1d, 20)?;
    println!("x_nearest_1d: {}", x_nearest_1d);
    // let x_linear_1d = interpolate_linear_1d(&input_1d, 10, Some(true))?;
    // println!("x_linear_1d: {}", x_linear_1d);
    // let input_2d = input.reshape((1, 1, 10, 10))?;
    // println!("input_2d: {}", input_2d);
    // let x_nearest_2d = interpolate_nearest_2d(&input_2d, (10, 10))?;
    // println!("x_nearest_2d: {}", x_nearest_2d);
    // let x_bilinear = interpolate_bilinear(&input_2d, (5, 5), Some(true), Some(false))?;
    // println!("x_bilinear: {}", x_bilinear);
    // let x_bicubic = interpolate_bicubic(&input_2d, (5, 5), Some(false), Some(true))?;
    // // let x_bicubic = interpolate_bicubic_standard(&input_2d, (5, 5), None)?;

    // println!("x_bicubic: {}", x_bicubic);
    Ok(())
}
