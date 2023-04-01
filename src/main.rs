#![feature(array_windows)]
#![feature(const_for)]
use std::{error::Error, time::Instant};

use dfdx::{data::*, optim::Adam, prelude::*, tensor::Cpu};
use indicatif::ProgressIterator;
use rand::{rngs::StdRng, SeedableRng};

const CTX_SIZE: usize = 24;
const CHAR_TENSOR_SIZE: usize = 256;
const CTX_TENSOR_SIZE: usize = CTX_SIZE * CHAR_TENSOR_SIZE;
const H1: usize = CTX_TENSOR_SIZE * 2;
const H2: usize = CTX_TENSOR_SIZE * 1;

const CHAR_TO_TENSOR_ARRAY: [[f32; CHAR_TENSOR_SIZE]; CHAR_TENSOR_SIZE] = generate_arrays();

type Mlp = (
    (Linear<CTX_TENSOR_SIZE, H1>, ReLU), // last n/256 characters
    (Linear<H1, H2>, ReLU),              // ???
    Linear<H2, CHAR_TENSOR_SIZE>,        // output as ascii
);

struct Dataset<A, I, F: Fn(&A, usize) -> I, FL: Fn(&A) -> usize> {
    data: A,
    getter: Box<F>,
    length: Box<FL>,
}

impl<A, I, F: Fn(&A, usize) -> I, FL: Fn(&A) -> usize> ExactSizeDataset for Dataset<A, I, F, FL> {
    type Item<'a> = I where Self: 'a;

    fn get(&self, index: usize) -> Self::Item<'_> {
        (self.getter)(&self.data, index)
    }

    fn len(&self) -> usize {
        (self.length)(&self.data)
    }
}

const BATCH_SIZE: usize = 32;

type Dev = Cuda;

fn main() -> Result<(), Box<dyn Error>> {
    // optimization
    dfdx::flush_denormals_to_zero();

    let dev = Dev::default();

    let mut rng = StdRng::seed_from_u64(0); // deterministic

    let mut model = dev.build_module::<Mlp, f32>();
    let mut grads = model.alloc_grads();
    let mut opt = Adam::new(&model, Default::default()); // adam opt with default params

    let data = include_str!("../1984.txt");
    let dataset = Dataset {
        data,
        getter: Box::new(|data: &&str, index| {
            let window = data
                .as_bytes()
                .array_windows::<CTX_SIZE>()
                .nth(index / CTX_SIZE)
                .unwrap()
                .map(char_to_array);
            let first_next = data
                .as_bytes()
                .array_windows::<CTX_SIZE>()
                .nth((index + 1) / CTX_SIZE)
                .unwrap()
                .first()
                .unwrap();
            (window, *first_next)
        }),
        length: Box::new(|data: &&str| data.len() - CTX_SIZE),
    };
    let tensorify = |(v, l): ([[f32; CHAR_TENSOR_SIZE]; CTX_SIZE], u8)| -> (
        Tensor<(Const<CTX_TENSOR_SIZE>,), f32, Dev>,
        Tensor<(Const<CHAR_TENSOR_SIZE>,), f32, Dev>,
    ) { (dev.tensor(v).reshape(), dev.tensor(char_to_array(l))) };

    for i_epoch in 0..50 {
        let mut total_epoch_loss = 0.;
        let mut num_batches = 0;

        let start = Instant::now();
        for (input, tag) in dataset
            .shuffled(&mut rng)
            .map(tensorify)
            .batch(Const::<BATCH_SIZE>)
            .collate()
            .stack()
            .progress()
        {
            let logits = model.forward_mut(input.traced(grads));
            let loss = cross_entropy_with_logits_loss(logits, tag);

            total_epoch_loss += loss.array();
            num_batches += 1;

            grads = loss.backward();
            opt.update(&mut model, &grads)?;
            model.zero_grads(&mut grads);
        }
        let time_taken = start.elapsed();

        println!(
            "Epoch {i_epoch} in {:?} ({:.3} batches/s): avg sample loss {:.5}",
            time_taken,
            num_batches as f32 / time_taken.as_secs_f32(),
            BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32,
        );
        let start_gen = Instant::now();
        println!(
            "Generating on epoch {i_epoch} (took {1:.2}s): {0}",
            generate("I went to the store and bought", 100, &mut model, &dev),
            start_gen.elapsed().as_secs_f32()
        );

        model.save("mnist.npz")?;
    }

    model.save("mnist.npz")?;

    let (mut pass, mut fail) = (0, 0);

    println!("guessed correctly {} out of {} times", pass, pass + fail);

    Ok(())
}

const fn char_to_array(ch: u8) -> [f32; CHAR_TENSOR_SIZE] {
    CHAR_TO_TENSOR_ARRAY[ch as usize]
}

fn max_idx<const A: usize>(x: &Tensor<(Const<A>,), f32, Dev>) -> usize {
    let arr = x.array();
    let (mut idx, mut max) = (0, arr[0]);
    for (i, x) in x.array().iter().enumerate().skip(1) {
        if x > &max {
            max = *x;
            idx = i;
        }
    }
    idx
}

const fn generate_arrays() -> [[f32; CHAR_TENSOR_SIZE]; CHAR_TENSOR_SIZE] {
    let mut res: [[f32; CHAR_TENSOR_SIZE]; CHAR_TENSOR_SIZE] =
        [[0.; CHAR_TENSOR_SIZE]; CHAR_TENSOR_SIZE];
    let mut x = 0; // workaround for non-const Iterator::next
    while x < CHAR_TENSOR_SIZE {
        res[x][x] = 1.;
        x += 1;
    }
    res
}

fn generate(
    x: &str,
    n: usize,
    model: &mut <Mlp as BuildOnDevice<Dev, f32>>::Built,
    dev: &Dev,
) -> String {
    let mut ctx_string = x.to_owned();

    let upper = ctx_string.len() - CTX_SIZE;
    let _ = ctx_string.drain(0..upper);

    let mut res = x.to_owned();
    let mut ctx = [0.; CTX_TENSOR_SIZE];
    for (i, c) in ctx_string.bytes().enumerate() {
        ctx[i * CHAR_TENSOR_SIZE + c as usize] = 1.;
    }

    let mut ctx_tensor: Tensor<(Const<CTX_TENSOR_SIZE>,), f32, Dev> = dev.tensor(ctx);
    for _ in 0..n {
        let out = model.forward_mut(ctx_tensor.clone());
        let new = tensor_to_char(&out);

        res.push(new);
        let mut new_ctx = ctx_tensor.array();
        for i in 0..(new_ctx.len() - CHAR_TENSOR_SIZE) {
            new_ctx[i] = new_ctx[i + CHAR_TENSOR_SIZE];
        }
        let new_array = char_to_array(new as u8);
        for i in 0..CHAR_TENSOR_SIZE {
            new_ctx[new_ctx.len() - CHAR_TENSOR_SIZE + i] = new_array[i];
        }
        ctx_tensor = dev.tensor(new_ctx);
    }

    res
}

fn tensor_to_char(t: &Tensor<(Const<256>,), f32, Dev>) -> char {
    closest_to_one(t) as u8 as char
}

fn closest_to_one<const A: usize>(t: &Tensor<(Const<A>,), f32, Dev>) -> usize {
    t.array()
        .iter()
        .enumerate()
        .fold((0usize, t.array()[0]), |x, a| {
            if (1. - a.1).abs() < (1. - x.1).abs() {
                (a.0, *a.1)
            } else {
                x
            }
        })
        .0
}

fn create_all_tensors(
    dev: &Dev,
) -> [Tensor<(Const<CHAR_TENSOR_SIZE>,), f32, Dev>; CHAR_TENSOR_SIZE] {
    (0..u8::MAX)
        .into_iter()
        .map(char_to_array)
        .map(|x| dev.tensor(x))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
