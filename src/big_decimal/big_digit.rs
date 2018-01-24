/// Composing element of `BigDecimal`.
pub type BigDigit = u32;
/// Internal type for computations. Doubled size of `BigDigit`.
pub type DoubleBigDigit = u64;

pub const BITS: usize = 32;
pub const BASE: DoubleBigDigit = 1 << BITS;
pub const MASK: DoubleBigDigit = BASE - 1;

#[inline]
fn get_hi(n: DoubleBigDigit) -> BigDigit {
    (n >> BITS) as BigDigit
}

#[inline]
fn get_lo(n: DoubleBigDigit) -> BigDigit {
    (n & MASK) as BigDigit
}

#[inline]
pub fn from_double_big_digit(n: DoubleBigDigit) -> (BigDigit, BigDigit) {
    (get_hi(n), get_lo(n))
}

#[inline]
pub fn to_double_big_digit(hi: BigDigit, lo: BigDigit) -> DoubleBigDigit {
    (lo as DoubleBigDigit) | ((hi as DoubleBigDigit) << BITS)
}
