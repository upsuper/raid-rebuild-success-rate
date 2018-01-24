# RAID rebuild success rate

A simple library with a WebAssembly-based web app for calculating the
success rate of RAID rebuild after one disk fails.

It solely considered the unrecoverable error (URE) which can occur when
reading large amount of data from disks. And the calculation is based
on the assumption that if you have more redundancy, a rebuild would only
fail if enough redundant disks fail at the same sector.

## Credits

This application uses code from other open source projects:
* [num-bigint](https://github.com/rust-num/num-bigint)
    * Copyright (c) 2014 The Rust Project Developers
    * License (MIT) https://github.com/rust-num/num-bigint/blob/master/LICENSE-MIT
