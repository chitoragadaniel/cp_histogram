use image::{GenericImageView, Pixel}; // Import the Pixel trait
use std::sync::{Arc, Mutex};
use std::time::{Instant};
use std::thread;

const HIST_SIZE: usize = 256;

type FineGrainedHistogram = [[Mutex<u32>; HIST_SIZE]; 3];
type Histogram = [[u32; HIST_SIZE]; 3]; // [R, G, B] histograms

fn image_to_rgb_array(image_path: &str) -> Vec<u32> {
    // Load the image
    let img = image::open(image_path).expect("Failed to open image");

    // Get image dimensions
    let (width, height) = img.dimensions();

    // Create a vector to hold the RGB values as 3-byte values
    let mut rgb_array = Vec::with_capacity((width * height) as usize);

    // Iterate over each pixel and convert it to the desired format
    for pixel in img.pixels() {
        let rgba = pixel.2.to_rgba(); // Now to_rgba is correctly in scope
        let r = rgba[0] as u32;
        let g = rgba[1] as u32;
        let b = rgba[2] as u32;

        // Combine RGB into a single 32-bit value, where the least significant byte is Blue,
        // the second least significant byte is Green, and the third least significant byte is Red
        let rgb_value = (r << 16) | (g << 8) | b;
        rgb_array.push(rgb_value);
    }

    rgb_array
}

fn generate_rgb_array(width: u32, height: u32) -> Vec<u32> {
    // Create a vector to hold the RGB values as 3-byte values
    let mut rgb_array = Vec::with_capacity((width * height) as usize);

    // Generate RGB values based on iteration index
    for i in 0..(width * height) {
        let r = ((i >> 16) & 0xFF) as u32; // Red component based on higher bits of i
        let g = ((i >> 8) & 0xFF) as u32;  // Green component based on middle bits of i
        let b = (i & 0xFF) as u32;         // Blue component based on lower bits of i

        // Combine RGB into a single 32-bit value, where the least significant byte is Blue,
        // the second least significant byte is Green, and the third least significant byte is Red
        let rgb_value = (r << 16) | (g << 8) | b;
        rgb_array.push(rgb_value);
    }

    rgb_array
}


fn hist_diff(h1: &Histogram, h2: &Histogram) -> bool {
    let mut diff_r = 0;
    let mut diff_g = 0;
    let mut diff_b = 0;

    for i in 0..HIST_SIZE {
        if h1[0][i] != h2[0][i] {
            diff_r += 1;
        }
        if h1[1][i] != h2[1][i] {
            diff_g += 1;
        }
        if h1[2][i] != h2[2][i] {
            diff_b += 1;
        }
    }

    // Print differences    
    if diff_r != 0 {
        println!("Error in the R channel: {}", diff_r);
    }
    if diff_g != 0 {
        println!("Error in the G channel: {}", diff_g);
    }
    if diff_b != 0 {
        println!("Error in the B channel: {}", diff_b);
    }


    diff_r != 0 || diff_g != 0 || diff_b != 0
}

fn impl_hist_seq(image_data: &[u32]) -> Histogram {
    let mut histogram = [[0u32; HIST_SIZE]; 3];

    for &pixel in image_data {
        // Extract the RGB components from the packed u32 value
        let r = ((pixel >> 16) & 0xFF) as usize; // Red component
        let g = ((pixel >> 8) & 0xFF) as usize;  // Green component
        let b = (pixel & 0xFF) as usize;         // Blue component

        // Increment the corresponding histogram bins
        histogram[0][r] += 1; // Red histogram
        histogram[1][g] += 1; // Green histogram
        histogram[2][b] += 1; // Blue histogram
    }

    histogram
}

fn impl_hist_ver1(image_data: &[u32], num_threads: usize) -> Histogram {
    // Initialize fine-grained histogram with Mutex<u32>
    let fine_grained_histogram: FineGrainedHistogram = [
        [(); HIST_SIZE].map(|_| Mutex::new(0)),
        [(); HIST_SIZE].map(|_| Mutex::new(0)),
        [(); HIST_SIZE].map(|_| Mutex::new(0)),
    ];

    let fine_grained_histogram = Arc::new(fine_grained_histogram);
    let image_data = Arc::new(image_data.to_vec()); // Clone the image_data into an Arc

    let chunk_size = (image_data.len() + num_threads - 1) / num_threads; // Ensure even distribution

    let mut handles = vec![];

    for i in 0..num_threads {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, image_data.len());
        let chunk = Arc::clone(&image_data);
        let histogram = Arc::clone(&fine_grained_histogram);

        let handle = thread::spawn(move || {
            let chunk = &chunk[start..end]; // Get the chunk for this thread
            for &pixel in chunk {
                let r = ((pixel >> 16) & 0xFF) as usize; // Red component
                let g = ((pixel >> 8) & 0xFF) as usize;  // Green component
                let b = (pixel & 0xFF) as usize;         // Blue component

                // Update histogram bins with fine-grained locking
                let mut red_lock = histogram[0][r].lock().unwrap();
                *red_lock += 1;
                let mut green_lock = histogram[1][g].lock().unwrap();
                *green_lock += 1;
                let mut blue_lock = histogram[2][b].lock().unwrap();
                *blue_lock += 1;
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Convert the fine-grained histogram to the desired Histogram type
    let mut histogram = [[0u32; HIST_SIZE]; 3];

    for i in 0..HIST_SIZE {
        histogram[0][i] = *fine_grained_histogram[0][i].lock().unwrap();
        histogram[1][i] = *fine_grained_histogram[1][i].lock().unwrap();
        histogram[2][i] = *fine_grained_histogram[2][i].lock().unwrap();
    }

    histogram
}

fn impl_hist_ver2(image_data: &[u32], num_threads: usize) -> Histogram {
    let image_data = Arc::new(image_data.to_vec()); // Clone image data into an Arc
    let mut global_histogram = [[0u32; HIST_SIZE]; 3];

    let chunk_size = (image_data.len() + num_threads - 1) / num_threads; // Ensure even distribution

    let mut handles = vec![];

    for i in 0..num_threads {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, image_data.len());
        let chunk = image_data.clone(); // Clone Arc to share with thread

        let handle = thread::spawn(move || {
            let chunk = &chunk[start..end]; // Get the chunk for this thread
            let mut local_histogram = [[0u32; HIST_SIZE]; 3];

            for &pixel in chunk {
                let r = ((pixel >> 16) & 0xFF) as usize; // Red component
                let g = ((pixel >> 8) & 0xFF) as usize;  // Green component
                let b = (pixel & 0xFF) as usize;         // Blue component

                // Update local histogram
                local_histogram[0][r] += 1; // Red histogram
                local_histogram[1][g] += 1; // Green histogram
                local_histogram[2][b] += 1; // Blue histogram
            }

            local_histogram
        });

        handles.push(handle);
    }

    for handle in handles {
        let local_histogram = handle.join().expect("Thread panicked");

        // Merge local histogram into the global histogram
        for i in 0..HIST_SIZE {
            global_histogram[0][i] += local_histogram[0][i];
            global_histogram[1][i] += local_histogram[1][i];
            global_histogram[2][i] += local_histogram[2][i];
        }
    }

    global_histogram
}

fn calc_img_histogram(image_path: &str) {
    let rgb_array = image_to_rgb_array(image_path);
    
    // Load the image
    let img = image::open(image_path).expect("Failed to open image");

    // Get image dimensions
    let (width, height) = img.dimensions();

    println!("Input: {:?}; Width: {:?}, Height: {:?}", image_path, width, height);

    let mut start = Instant::now();
    let hist_seq = impl_hist_seq(&rgb_array);
    let mut elapsed = start.elapsed();
    println!("Kernel took: {:?}s", elapsed);

    start = Instant::now();
    let hist_ver1 = impl_hist_ver1(&rgb_array, 10);
    elapsed = start.elapsed();
    println!("Kernel took: {:?}s", elapsed);

    start = Instant::now();
    let hist_ver2 = impl_hist_ver2(&rgb_array, 10); // Tested on 720p, 1080p and 4k images and overall at 10 threads the performances peaks
    elapsed = start.elapsed();
    println!("Kernel took: {:?}s", elapsed);

    if hist_diff(&hist_seq, &hist_ver1) {println!("Error in histogram -- chech ver1!")};
    if hist_diff(&hist_seq, &hist_ver2) {println!("Error in histogram -- chech ver2!")};
    println!();
}

fn calc_gen_img_histogram(height: u32, width: u32) {
    let rgb_array = generate_rgb_array(height, width);
    println!("Input generated image; Width: {:?}, Height: {:?}", width, height);

    let mut start = Instant::now();
    let hist_seq = impl_hist_seq(&rgb_array);
    let mut elapsed = start.elapsed();
    println!("Kernel took: {:?}s", elapsed);

    start = Instant::now();
    let hist_ver1 = impl_hist_ver1(&rgb_array, 10);
    elapsed = start.elapsed();
    println!("Kernel took: {:?}s", elapsed);

    start = Instant::now();
    let hist_ver2 = impl_hist_ver2(&rgb_array, 10); // Tested on 720p, 1080p and 4k images and overall at 10 threads the performances peaks
    elapsed = start.elapsed();
    println!("Kernel took: {:?}s", elapsed);

    if hist_diff(&hist_seq, &hist_ver1) {println!("Error in histogram -- chech ver1!")};
    if hist_diff(&hist_seq, &hist_ver2) {println!("Error in histogram -- chech ver2!")};
    println!();
}

fn main() {
    calc_img_histogram("../img_1.jpeg");
    calc_img_histogram("../img_2.jpeg");
    calc_gen_img_histogram(10000, 10000);
}
