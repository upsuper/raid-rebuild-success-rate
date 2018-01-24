extern crate raid_rebuild_success_rate;

use raid_rebuild_success_rate::{Config, compute};

fn print_result(config: &Config) {
    let result = compute(config);
    let mut iter = result.iter().enumerate();
    println!("No redundancy: {:.2}%", iter.next().unwrap().1 * 100.);
    println!("1 disk redundancy: {:.2}%", iter.next().unwrap().1 * 100.);
    for (i, rate) in iter {
        println!("{} disks redundancy: {:.2}%", i, rate * 100.);
    }
}

fn main() {
    println!("1 TB * 4:");
    print_result(&Config {
        sector_size: 4096,
        sectors: 1_000_000_000_000 / 4096,
        disks: 4,
        ure_rate_exp: 14,
    });

    println!();
    println!("10 TB * 8:");
    print_result(&Config {
        sector_size: 4096,
        sectors: 10_000_000_000_000 / 4096,
        disks: 8,
        ure_rate_exp: 14,
    });
}
