#![no_std]

#[cfg(test)]
#[macro_use]
extern crate std;

extern crate num_traits;

mod big_decimal;

use big_decimal::{BigDecimal, pow};
use num_traits::{One, Zero};

/// The RAID configuration
pub struct Config {
    /// Size of each sector in bytes. "Advanced Format" means 4096 bytes in
    /// general, otherwise it is usually 512 bytes.
    pub sector_size: u64,
    /// Number of total sectors. Basically capacity / sector size.
    pub sectors: u64,
    /// Number of disks in total.
    pub disks: u32,
    /// Negative exponent of rate of unrecoverable read errors per bit read.
    /// For "< 1 in 10^14", this should be the "14" part.
    pub ure_rate_exp: u8,
}

/// Given the configuration of the RAID, return an array of success rates
/// for rebuild after one disk fails. Each item means the rate for number
/// of redundant disks. NaN for those don't apply.
pub fn compute(config: &Config) -> [f64; 4] {
    const BITS_PER_BYTE: u64 = 8;

    let ure_rate = {
        let mut result = BigDecimal::one();
        for _ in 0..config.ure_rate_exp {
            result /= 10;
        }
        result
    };
    let safe_bit_rate = BigDecimal::one() - ure_rate;
    let safe_sector_rate = pow(safe_bit_rate, config.sector_size as u64 * BITS_PER_BYTE);
    let ure_sector_rate = BigDecimal::one() - safe_sector_rate;

    let mut result = [core::f64::NAN; 4];
    result[0] = 0.;
    let mut rate = BigDecimal::zero();
    for (i, result) in result.iter_mut().enumerate().skip(1) {
        if config.disks - 1 < i as u32 {
            break;
        }
        // At most m out of n disks can fail without failing the rebuild
        // after one disk has already failed.
        let n = config.disks - 1;
        let m = i as u32 - 1;
        let factor = ((n - m + 1)..(n + 1)).product::<u32>() / (1..(m + 1)).product::<u32>();
        rate += pow(safe_sector_rate, n as u64 - m as u64) *
            pow(ure_sector_rate, m as u64) * factor;
        let high_prec_result = pow(rate.clone(), config.sectors);
        *result = high_prec_result.into();
    }

    result
}
