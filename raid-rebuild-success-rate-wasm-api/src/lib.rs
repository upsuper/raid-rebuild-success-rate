#![no_std]

extern crate raid_rebuild_success_rate;

extern "C" {
    fn report_result(one_disk: f64, two_disks: f64, three_disks: f64);
}

#[no_mangle]
pub extern "C" fn compute_raid_rebuild_success_rate(
    single_disk_size: f64,
    sector_size: u32,
    disks: u32,
    ure_rate_exp: u8,
) {
    let config = raid_rebuild_success_rate::Config {
        sector_size: sector_size as u64,
        sectors: (single_disk_size / sector_size as f64).round() as u64,
        disks: disks,
        ure_rate_exp: ure_rate_exp,
    };
    let result = raid_rebuild_success_rate::compute(&config);
    unsafe { report_result(result[1], result[2], result[3]); }
}
