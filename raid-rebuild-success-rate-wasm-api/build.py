#!/usr/bin/env python3

import json
import re
import shutil
import subprocess

from pathlib import Path

def execute(args):
    process = subprocess.run(args, stdout=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError("Failed to execute: " + " ".join(args))
    return process.stdout

cargo_result = execute([
    "cargo", "+beta", "build",
    "--target=wasm32-unknown-unknown",
    "--release",
    "--message-format=json",
]).splitlines()[-1]
cargo_result = json.loads(cargo_result)
source_wasm = cargo_result["filenames"][0]

execute(["wasm-gc", source_wasm])

SHOULD_SNIP = [
    re.compile(rb".*core::fmt::"),
    re.compile(rb".*core::panicking::"),
    re.compile(rb".*std::"),
    re.compile(rb"dlmalloc::"),
]
symbols = execute(["wasm-nm", "-p", "-j", source_wasm]).splitlines()
snip_functions = set()
for snip in SHOULD_SNIP:
    snip_functions.update(filter(lambda f: re.match(snip, f), symbols))
target = Path() / "ui" / "raid-rebuild-success-rate.wasm";
execute(["wasm-snip", "-o", target, source_wasm, *snip_functions])

execute(["wasm-gc", target])

opt_target = target.with_suffix(".opt" + target.suffix)
execute([
    "wasm-opt", "-O3", "-Oz",
    "--duplicate-function-elimination",
    "-o", opt_target,
    target
])
