let memory = new WebAssembly.Memory({initial: 1});
let output = new Float64Array(memory.buffer, 0, 4);
let compute_raid_rebuild_success_rate;

function $i(id) {
  return document.getElementById(id);
}

async function loadWasm() {
  let resp = await fetch("raid-rebuild-success-rate.opt.wasm");
  let bytes = await resp.arrayBuffer();
  let importObjs = {
    env: {
      round: Math.round,
      report_result: function(one_disk, two_disks, three_disks) {
        function output_result(id, result) {
          if (isNaN(result)) {
            result = "n/a";
          } else {
            result = (result * 100).toFixed(2) + "%";
          }
          $i(id).textContent = result;
        }
        output_result("one-disk", one_disk);
        output_result("two-disk", two_disks);
        output_result("three-disk", three_disks);
        $i("output").hidden = false;
      }
    }
  };
  let results = await WebAssembly.instantiate(bytes, importObjs);
  compute_raid_rebuild_success_rate =
    results.instance.exports.compute_raid_rebuild_success_rate;
}
loadWasm();

$i("compute").addEventListener("click", function() {
  let single_disk_size = parseFloat($i("disk_size").value) * 1e12;
  let sector_size = parseInt($i("sector_size").value);
  let disks = parseInt($i("disks").value);
  let ure_rate_exp = parseInt($i("ure_rate_exp").value);
  compute_raid_rebuild_success_rate(
    single_disk_size, sector_size, disks, ure_rate_exp, output);
});
