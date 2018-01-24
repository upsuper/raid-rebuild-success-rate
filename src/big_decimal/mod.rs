mod big_digit;

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use core::str;
use num_traits::{One, Zero};
use self::big_digit::{DoubleBigDigit, from_double_big_digit, to_double_big_digit};

pub use self::big_digit::BigDigit;

const INTEGER_PART: usize = 1;
const FRACTIONAL_PART: usize = 31;
const TOTAL_LENGTH: usize = INTEGER_PART + FRACTIONAL_PART;

/// A fixed-point decimal implementation.
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct BigDecimal([BigDigit; INTEGER_PART + FRACTIONAL_PART]);

impl BigDecimal {
    /// Returns the minimum difference between two numbers in this type.
    pub fn epsilon() -> Self {
        let mut result = Self::zero();
        result.0[0] = 1;
        result
    }

    /// Returns the maximum value of this type.
    pub fn max_value() -> Self {
        BigDecimal([BigDigit::max_value(); TOTAL_LENGTH])
    }

    /// Returns the minimum value of this type, which is zero.
    pub fn min_value() -> Self {
        Self::zero()
    }

    #[cfg(test)]
    fn digit(&self, pos: isize) -> &BigDigit {
        &self.0[(pos + FRACTIONAL_PART as isize) as usize]
    }

    #[cfg(test)]
    fn digit_mut(&mut self, pos: isize) -> &mut BigDigit {
        &mut self.0[(pos + FRACTIONAL_PART as isize) as usize]
    }
}

impl Default for BigDecimal {
    fn default() -> BigDecimal {
        Self::zero()
    }
}

fn is_zero(digits: &[BigDigit]) -> bool {
    digits.iter().all(|n| *n == 0)
}

impl Zero for BigDecimal {
    fn zero() -> BigDecimal {
        BigDecimal([0; TOTAL_LENGTH])
    }

    fn is_zero(&self) -> bool {
        is_zero(&self.0)
    }
}

impl One for BigDecimal {
    fn one() -> BigDecimal {
        let mut result = Self::zero();
        result.0[FRACTIONAL_PART] = 1;
        result
    }
}

/// Adds one epsilon to the number and returns whether it overflows.
fn add_epsilon(num: &mut BigDecimal) -> bool {
    let mut carry = 1;
    for n in num.0.iter_mut() {
        *n = add_with_carry(*n, 0, &mut carry);
    }
    return carry != 0;
}

impl From<f64> for BigDecimal {
    fn from(from: f64) -> BigDecimal {
        assert!(from >= 0., "attempt to convert a negative float");
        const BASE: f64 = big_digit::BASE as f64;
        let mut result = BigDecimal::zero();
        // Fill the integer part.
        let mut integer = from.trunc();
        for r in result.0[FRACTIONAL_PART..].iter_mut() {
            *r = (integer % BASE) as BigDigit;
            integer = (integer / BASE).trunc();
        }
        assert!(integer == 0., "attempt to convert with overflow");
        // Fill the fractional part.
        let mut fract = from.fract();
        for r in result.0[..FRACTIONAL_PART].iter_mut().rev() {
            fract *= BASE;
            *r = fract.trunc() as BigDigit;
            fract = fract.fract();
        }
        // Round the number if necessary.
        if fract >= 0.5 {
            if add_epsilon(&mut result) {
                panic!("attempt to convert with overflow");
            }
        }
        result
    }
}

impl<'a> From<&'a BigDecimal> for f64 {
    fn from(from: &BigDecimal) -> f64 {
        const BASE: f64 = big_digit::BASE as f64;
        let mut base = BASE.powi(INTEGER_PART as i32 - 1);
        let mut result = 0.;
        for n in from.0.iter().rev() {
            result += *n as f64 * base;
            base /= BASE;
        }
        result
    }
}

impl From<BigDecimal> for f64 {
    fn from(from: BigDecimal) -> f64 {
        f64::from(&from)
    }
}

fn add_with_carry(l: BigDigit, r: BigDigit, carry: &mut BigDigit) -> BigDigit {
    let (hi, lo) = from_double_big_digit(l as DoubleBigDigit + r as DoubleBigDigit + (*carry as DoubleBigDigit));
    *carry = hi;
    lo
}

impl<'a> Add<&'a BigDecimal> for BigDecimal {
    type Output = Self;
    fn add(mut self, rhs: &BigDecimal) -> Self {
        self += rhs;
        self
    }
}

impl<'a> AddAssign<&'a BigDecimal> for BigDecimal {
    fn add_assign(&mut self, rhs: &BigDecimal) {
        let mut carry = 0;
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a = add_with_carry(*a, *b, &mut carry);
        }
        assert!(carry == 0, "attempt to add with overflow");
    }
}

fn sub_with_borrow(l: BigDigit, r: BigDigit, borrow: &mut BigDigit) -> BigDigit {
    let l = l as DoubleBigDigit;
    let r = r as DoubleBigDigit + *borrow as DoubleBigDigit;
    if l >= r {
        *borrow = 0;
        (l - r) as BigDigit
    } else {
        *borrow = 1;
        (big_digit::BASE + l - r) as BigDigit
    }
}

impl Add<BigDecimal> for BigDecimal {
    type Output = Self;
    fn add(self, rhs: BigDecimal) -> Self {
        self + &rhs
    }
}

impl AddAssign<BigDecimal> for BigDecimal {
    fn add_assign(&mut self, rhs: BigDecimal) {
        *self += &rhs;
    }
}

impl<'a> Sub<&'a BigDecimal> for BigDecimal {
    type Output = Self;
    fn sub(mut self, rhs: &BigDecimal) -> Self {
        self -= rhs;
        self
    }
}

impl<'a> SubAssign<&'a BigDecimal> for BigDecimal {
    fn sub_assign(&mut self, rhs: &BigDecimal) {
        let mut borrow = 0;
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a = sub_with_borrow(*a, *b, &mut borrow);
        }
        assert!(borrow == 0, "attempt to subtract with overflow");
    }
}

impl Sub<BigDecimal> for BigDecimal {
    type Output = Self;
    fn sub(self, rhs: BigDecimal) -> Self {
        self - &rhs
    }
}

impl SubAssign<BigDecimal> for BigDecimal {
    fn sub_assign(&mut self, rhs: BigDecimal) {
        *self -= &rhs;
    }
}

fn mul_with_carry(l: BigDigit, r: BigDigit, carry: &mut BigDigit) -> BigDigit {
    let (hi, lo) = from_double_big_digit(l as DoubleBigDigit * r as DoubleBigDigit + *carry as DoubleBigDigit);
    *carry = hi;
    lo
}

/// Multiplies `l` with `r`, stores the result into `l`, and returns
/// carry when `l` is not enough to hold the result.
fn mul_digit(l: &mut [BigDigit], r: BigDigit) -> BigDigit {
    let mut carry = 0;
    for n in l.iter_mut() {
        *n = mul_with_carry(*n, r, &mut carry);
    }
    carry
}

impl Mul<BigDigit> for BigDecimal {
    type Output = BigDecimal;
    fn mul(mut self, rhs: BigDigit) -> BigDecimal {
        self *= rhs;
        self
    }
}

impl MulAssign<BigDigit> for BigDecimal {
    fn mul_assign(&mut self, rhs: BigDigit) {
        let carry = mul_digit(&mut self.0, rhs);
        assert!(carry == 0, "attempt to multiple with overflow");
    }
}

fn mul_add_acc_with_carry(acc: BigDigit, l: BigDigit, r: BigDigit, carry: &mut BigDigit) -> BigDigit {
    let (hi, lo) = from_double_big_digit(acc as DoubleBigDigit +
                                         l as DoubleBigDigit * r as DoubleBigDigit +
                                         *carry as DoubleBigDigit);
    *carry = hi;
    lo
}

fn mul_add_acc(acc: &mut [BigDigit], l: &[BigDigit], r: BigDigit) {
    if r == 0 {
        return;
    }
    let mut carry = 0;
    let (a_lo, a_hi) = acc.split_at_mut(l.len());
    for (acc, l) in a_lo.iter_mut().zip(l) {
        *acc = mul_add_acc_with_carry(*acc, *l, r, &mut carry);
    }
    let mut a = a_hi.iter_mut();
    while carry != 0 {
        let a = a.next().expect("carry overflow during multiplication");
        *a = add_with_carry(*a, 0, &mut carry);
    }
    
}

impl<'a, 'b> Mul<&'b BigDecimal> for &'a BigDecimal {
    type Output = BigDecimal;

    fn mul(self, rhs: &BigDecimal) -> BigDecimal {
        let mut acc = [0; TOTAL_LENGTH * 2 + 1];
        for (i, x) in self.0.iter().enumerate() {
            mul_add_acc(&mut acc[i..], &rhs.0, *x);
        }
        // Any non-zero value in extra integer part is overflow.
        for x in acc[FRACTIONAL_PART * 2 + INTEGER_PART..].iter() {
            assert!(*x == 0, "attempt to multiple with overflow");
        }
        // Round the fractional part.
        if acc[FRACTIONAL_PART - 1] >= 1 << (big_digit::BITS - 1) {
            let mut carry = 1;
            for a in acc[FRACTIONAL_PART..].iter_mut() {
                *a = add_with_carry(*a, 0, &mut carry);
                if carry == 0 {
                    break;
                }
            }
        }
        let mut result = BigDecimal::zero();
        result.0.copy_from_slice(&acc[FRACTIONAL_PART..FRACTIONAL_PART + TOTAL_LENGTH]);
        result
    }
}

impl<'a> MulAssign<&'a BigDecimal> for BigDecimal {
    fn mul_assign(&mut self, rhs: &BigDecimal) {
        *self = &*self * rhs;
    }
}

impl Mul<BigDecimal> for BigDecimal {
    type Output = BigDecimal;
    fn mul(self, rhs: BigDecimal) -> BigDecimal {
        &self * &rhs
    }
}

impl MulAssign<BigDecimal> for BigDecimal {
    fn mul_assign(&mut self, rhs: BigDecimal) {
        *self *= &rhs;
    }
}

/// `dividend` divided by `divisor`, the quotient is stored in `dividend`, and
/// the remainder is returned from the function.
fn div_rem_digit(dividend: &mut [BigDigit], divisor: BigDigit) -> BigDigit {
    let mut rem = 0;
    let rhs = divisor as DoubleBigDigit;
    for d in dividend.iter_mut().rev() {
        let lhs = to_double_big_digit(rem, *d);
        *d = (lhs / rhs) as BigDigit;
        rem = (lhs % rhs) as BigDigit;
    }
    rem
}

impl Div<BigDigit> for BigDecimal {
    type Output = BigDecimal;
    fn div(mut self, rhs: BigDigit) -> BigDecimal {
        self /= rhs;
        self
    }
}

impl DivAssign<BigDigit> for BigDecimal {
    fn div_assign(&mut self, rhs: BigDigit) {
        let rem = div_rem_digit(&mut self.0, rhs);
        let next_digit = to_double_big_digit(rem, 0) / rhs as DoubleBigDigit;
        if next_digit >= 1 << (big_digit::BITS - 1) {
            if add_epsilon(self) {
                unreachable!("divide with overflow?");
            }
        }
    }
}

impl fmt::Display for BigDecimal {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        const BASE: BigDigit = 1_000_000_000;
        // Maximum number of digits for remainder of `BASE`.
        const DIGIT_PER_BASE: usize = 9;
        // Each `BigDigit` can convert to at most 10 decimal digits.
        const DIGIT_PER_BIG_DIGIT: usize = 10;

        let write_digits = |mut num: BigDigit, buf: &mut [u8]| {
            debug_assert!(num < BASE);
            for c in buf.iter_mut().rev() {
                *c = b'0' + (num % 10) as u8;
                num /= 10;
            }
            debug_assert!(num == 0);
        };

        // Write the integer part.
        if is_zero(&self.0[FRACTIONAL_PART..]) {
            f.write_str("0")?;
        } else {
            let mut int_part = [0; INTEGER_PART];
            int_part.copy_from_slice(&self.0[FRACTIONAL_PART..]);
            let mut int_part_str = [b'0'; INTEGER_PART * DIGIT_PER_BIG_DIGIT];
            for s in int_part_str.chunks_mut(DIGIT_PER_BASE).rev() {
                let mut rem = div_rem_digit(&mut int_part, BASE);
                write_digits(rem, s);
            }
            let first_non_zero = int_part_str.iter().position(|x| *x != b'0').unwrap();
            f.write_str(str::from_utf8(&int_part_str[first_non_zero..]).unwrap())?;
        }

        let prec = f.precision();
        if (prec.is_none() && is_zero(&self.0[..FRACTIONAL_PART])) || prec == Some(0) {
            return Ok(());
        }
        f.write_str(".")?;

        // Write the fractional part.
        let max_dec_digits = ((big_digit::BASE as f64).log10() * FRACTIONAL_PART as f64).round() as usize;
        let mut left_digits = prec.unwrap_or(max_dec_digits);
        let mut frac_part = [0; FRACTIONAL_PART + 1];
        frac_part[..FRACTIONAL_PART].copy_from_slice(&self.0[..FRACTIONAL_PART]);
        let mut frac_str = [0; DIGIT_PER_BASE];
        while !is_zero(&frac_part) || prec.is_some() {
            let carry = mul_digit(&mut frac_part, BASE);
            debug_assert!(carry == 0);
            write_digits(frac_part[FRACTIONAL_PART], &mut frac_str);
            if left_digits >= DIGIT_PER_BASE {
                f.write_str(str::from_utf8(&frac_str).unwrap())?;
                frac_part[FRACTIONAL_PART] = 0;
                left_digits -= DIGIT_PER_BASE;
            } else {
                f.write_str(str::from_utf8(&frac_str[..left_digits]).unwrap())?;
                break;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for BigDecimal {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fn write_digit(f: &mut fmt::Formatter, digit: &BigDigit) -> Result<(), fmt::Error> {
            write!(f, "{1:00$X}", big_digit::BITS / 4, digit)
        }
        let mut first = true;
        for digit in self.0[FRACTIONAL_PART..].iter().rev() {
            if !first {
                f.write_str("_")?;
            } else {
                first = false;
            }
            write_digit(f, digit)?;
        }
        f.write_str(".")?;
        let mut first = true;
        for digit in self.0[..FRACTIONAL_PART].iter().rev() {
            if !first {
                f.write_str("_")?;
            } else {
                first = false;
            }
            write_digit(f, digit)?;
        }
        Ok(())
    }
}

/// Raises a value to the power of exp, using exponentiation by squaring.
///
/// Note: this is copied from `num_traits::pow`. The only difference is the type
/// of exponent, which is `u64` here but `usize` in num-traits.
pub fn pow<T: Clone + One + Mul<T, Output = T>>(mut base: T, mut exp: u64) -> T {
    if exp == 0 { return T::one() }

    while exp & 1 == 0 {
        base = base.clone() * base;
        exp >>= 1;
    }
    if exp == 1 { return base }

    let mut acc = base.clone();
    while exp > 1 {
        exp >>= 1;
        base = base.clone() * base;
        if exp & 1 == 1 {
            acc = acc * base.clone();
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use std::prelude::v1::*;

    use num_traits::{Zero, One};
    use super::{BigDecimal, BigDigit, FRACTIONAL_PART, INTEGER_PART, TOTAL_LENGTH, big_digit, pow};

    fn build_big_decimal<I>(nums: I) -> BigDecimal
        where I: IntoIterator<Item=(isize, BigDigit)>
    {
        let mut result = BigDecimal::zero();
        for (pos, num) in nums.into_iter() {
            *result.digit_mut(pos) = num;
        }
        result
    }

    #[test]
    fn test_from_f64() {
        assert!(BigDecimal::from(0.0f64) == BigDecimal::zero());
        assert!(BigDecimal::from(1.0f64) == BigDecimal::one());
        assert!(BigDecimal::from(0.5f64) == build_big_decimal(vec![(-1, 1 << 31)]));
        assert!(BigDecimal::from(1.5f64) == build_big_decimal(vec![(0, 1), (-1, 1 << 31)]));
    }

    #[test]
    #[should_panic]
    fn test_from_f64_overflow() {
        BigDecimal::from(2.0f64.powi((INTEGER_PART * big_digit::BITS) as i32));
    }

    #[test]
    fn test_into_f64() {
        assert!(f64::from(&BigDecimal::zero()) == 0.0f64);
        assert!(f64::from(&BigDecimal::one()) == 1.0f64);
        for x in 0..100000 {
            let x = x as f64 / 100.;
            assert!(f64::from(&BigDecimal::from(x)) == x);
        }
    }

    #[test]
    fn test_add() {
        assert!(BigDecimal::one() + BigDecimal::zero() == BigDecimal::one());
        assert!(BigDecimal::zero() + BigDecimal::epsilon() == BigDecimal::epsilon());
        assert!(BigDecimal::one() + BigDecimal::one() == build_big_decimal(vec![(0, 2)]));
        assert!(BigDecimal::one() + BigDecimal::epsilon() ==
                build_big_decimal(vec![(0, 1), (-(FRACTIONAL_PART as isize), 1)]));
    }

    #[test]
    #[should_panic]
    fn test_add_overflow() {
        BigDecimal::max_value() + BigDecimal::epsilon();
    }

    #[test]
    fn test_sub() {
        assert!(BigDecimal::one() - BigDecimal::one() == BigDecimal::zero());
        assert!(BigDecimal::one() - BigDecimal::epsilon() ==
                build_big_decimal((0..FRACTIONAL_PART).map(|n| {
                    (-1 - n as isize, BigDigit::max_value())
                })));
    }

    #[test]
    #[should_panic]
    fn test_sub_overflow() {
        BigDecimal::min_value() - BigDecimal::epsilon();
    }

    #[test]
    fn test_mul_digit() {
        assert!(BigDecimal::one() * 10 == build_big_decimal(vec![(0, 10)]));
        assert!(BigDecimal::zero() * BigDigit::max_value() == BigDecimal::zero());
        assert!(BigDecimal::epsilon() * BigDigit::max_value() ==
                build_big_decimal(vec![(-(FRACTIONAL_PART as isize), BigDigit::max_value())]));
    }

    #[test]
    #[should_panic]
    fn test_mul_digit_overflow() {
        BigDecimal::max_value() * 2;
    }

    #[test]
    fn test_mul() {
        assert!(BigDecimal::one() * BigDecimal::one() == BigDecimal::one());
        assert!(BigDecimal::one() * BigDecimal::epsilon() == BigDecimal::epsilon());
        assert!(BigDecimal::zero() * BigDecimal::max_value() == BigDecimal::zero());
        assert!(BigDecimal::max_value() * BigDecimal::epsilon() ==
                build_big_decimal(vec![(-(FRACTIONAL_PART as isize) + 1, 1)]),
                "should be rounding up here");
        let one_sub_epsilon = BigDecimal::one() - BigDecimal::epsilon();
        assert!(one_sub_epsilon * one_sub_epsilon == one_sub_epsilon - BigDecimal::epsilon(),
                "0xffff...ffff * 0xffff...ffff = 0xffff...fffe000...001");
    }

    #[test]
    #[should_panic]
    fn test_mul_overflow() {
        BigDecimal::max_value() * (BigDecimal::one() + BigDecimal::epsilon());
    }

    #[test]
    fn test_div_digit() {
        assert!(BigDecimal::one() / 2 == build_big_decimal(vec![(-1, 1 << (big_digit::BITS - 1))]));
        assert!(BigDecimal::max_value() / BigDigit::max_value() == BigDecimal([1; TOTAL_LENGTH]));
        assert!((BigDecimal::one() - BigDecimal::epsilon()) / 2 == BigDecimal::one() / 2,
                "should be rounding up here");
    }
}
