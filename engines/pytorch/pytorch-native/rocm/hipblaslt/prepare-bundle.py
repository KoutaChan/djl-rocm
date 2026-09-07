#!/usr/bin/env python3
"""Build and stage the matching pinned hipBLASLt host runtime, without changing the SDK."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
PROFILES = json.loads((ROOT / "profiles.json").read_text())
MSGPACK_REVISION = "8c602e8579c7e7d65d6f9c6703c9699db3fb0488"
FMT_REVISION = "9cf9f38eded63e5e0fb95cd536ba51be601d7fa2"
YAML_REVISION = "f7320141120f720aecc4c32be25586e7da9eb978"


def run(*args, capture=False, env=None, stdin=None):
    return subprocess.run(args, check=True, text=True, env=env, input=stdin,
                          stdout=subprocess.PIPE if capture else None).stdout


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def properties(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def write_properties(path, values):
    path.write_text("".join(f"{key}={value}\n" for key, value in sorted(values.items())))


def find_sdk():
    for variable in ("ROCM_PATH", "ROCM_HOME"):
        if os.environ.get(variable):
            return Path(os.environ[variable]).resolve()
    if shutil.which("rocm-sdk"):
        return Path(run("rocm-sdk", "path", "--root", capture=True).strip()).resolve()
    sdk = Path("/opt/rocm")
    if (sdk / "bin/hipcc").is_file():
        return sdk.resolve()
    raise RuntimeError("Set ROCM_PATH to the matching ROCm development SDK")


def sdk_profile(sdk, flavor=None):
    match = re.match(r"\d+\.\d+\.\d+", (sdk / ".info/version").read_text().strip())
    if match is None:
        raise RuntimeError(f"Invalid ROCm SDK version: {sdk}")
    version = match[0]
    actual_flavor = "rocm" + ".".join(version.split(".")[:2])
    if flavor is not None and flavor != actual_flavor:
        raise RuntimeError(f"Requested {flavor}, but {sdk} contains {actual_flavor}")
    profile = PROFILES.get(actual_flavor, {}).get("sdk_versions", {}).get(version)
    if profile is None:
        raise RuntimeError(f"No pinned hipBLASLt profile for ROCm SDK {version}")
    header = (sdk / "include/hipblaslt/hipblaslt-version.h").read_text()
    for suffix, expected in zip(("MAJOR", "MINOR", "PATCH"), profile["hipblaslt_version"].split(".")):
        found = re.search(rf"#define\s+HIPBLASLT_VERSION_{suffix}\s+(\S+)", header)
        if found is None or found[1] != expected:
            raise RuntimeError(f"SDK {version} requires hipBLASLt {profile['hipblaslt_version']}")
    tweak = re.search(r"#define\s+HIPBLASLT_VERSION_TWEAK\s+(\S+)", header)
    if tweak is None or not re.fullmatch(r"[0-9a-f]{8,40}", tweak[1]) or not profile["source_revision"].startswith(tweak[1]):
        raise RuntimeError(f"SDK {version} hipBLASLt source differs from {profile['source_revision']}")
    return actual_flavor, version, profile


def sdk_metadata(sdk, requested_targets=None):
    directory = sdk / "lib/hipblaslt/library"
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name.endswith((".dat", ".dat.zlib")))
    targets = sorted(set(target for path in files for target in re.findall(r"gfx[0-9a-f]+", path.relative_to(directory).as_posix())))
    if requested_targets:
        requested = sorted(set(requested_targets.split(";")))
        if not set(requested) <= set(targets):
            raise RuntimeError(f"SDK Tensile metadata does not support requested GPU targets: {requested}")
        targets = requested
        # Global mapping / extension metadata also belongs to an architecture's runtime.
        files = [path for path in files if not re.findall(r"gfx[0-9a-f]+", path.relative_to(directory).as_posix())
                 or set(re.findall(r"gfx[0-9a-f]+", path.relative_to(directory).as_posix())) & set(targets)]
    if not targets:
        raise RuntimeError(f"Missing SDK Tensile metadata: {directory}")
    return targets, {path.relative_to(sdk).as_posix(): sha256(path) for path in files}


def source_patterns(profile):
    prefix = "" if profile["source_dir"] == "." else profile["source_dir"] + "/"
    directories = ["cmake", "library"]
    if profile["build_family"] == "legacy":
        directories += ["tensilelite/Tensile/Source", "tensilelite/rocisa"]
    else:
        directories += ["device-library", "tensilelite/cmake", "tensilelite/include", "tensilelite/src", "tensilelite/rocisa", "tensilelite/tests"]
    patterns = ["/*", "!/*/"]
    if prefix:
        patterns += ["/projects/", "!/projects/*/", "/projects/hipblaslt/", "!/projects/hipblaslt/*/"]
    patterns += [f"/{prefix}{name}/" for name in directories]
    # Exclude multi-gigabyte generated kernel logic from a host-only checkout.
    patterns += [f"!/{prefix}library/src/amd_detail/rocblaslt/src/Tensile/Logic/"]
    if profile["build_family"] == "modern":
        patterns += [f"/{prefix}tensilelite/*", f"!/{prefix}tensilelite/*/"]
        patterns += [f"/{prefix}{name}/" for name in directories if name.startswith("tensilelite/")]
        patterns += ["/shared/", "!/shared/*/", "/shared/origami/"]
    return patterns


def checkout_source(source, profile):
    revision = profile["source_revision"]
    if not (source / ".git").is_dir():
        run("git", "init", str(source))
    # Persist promisor configuration before checkout; missing blobs must not
    # trigger an unfiltered fetch of the entire rocm-libraries repository.
    for key, value in {
        "remote.origin.url": f"https://github.com/{profile['source_repository']}.git",
        "remote.origin.promisor": "true", "remote.origin.partialclonefilter": "blob:none",
        "extensions.partialClone": "origin",
    }.items():
        run("git", "-C", str(source), "config", key, value)
    run("git", "-C", str(source), "sparse-checkout", "init", "--no-cone")
    run("git", "-C", str(source), "sparse-checkout", "set", "--no-cone", "--stdin", stdin="\n".join(source_patterns(profile)) + "\n")
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, capture_output=True)
    if head.returncode or head.stdout.strip() != revision:
        run("git", "-C", str(source), "fetch", "--depth", "1", "--filter=blob:none", "origin", revision)
        run("git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD")


def apply_patches(source, profile):
    for name in profile["patches"]:
        command = ["git", "-C", str(source), "apply"]
        if name != "0001-bound-tensile-solution-cache.patch" and profile["source_dir"] != ".":
            command += ["--directory=" + profile["source_dir"]]
        patch = str(ROOT / "patches" / name)
        applied = subprocess.run(command + ["--reverse", "--check", patch], capture_output=True)
        if applied.returncode:
            run(*command, "--check", patch)
            run(*command, patch)


def public_symbols(library):
    symbols = set()
    for line in run("readelf", "--dyn-syms", "--wide", str(library), capture=True).splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[6] != "UND":
            name = fields[7].split("@", 1)[0]
            if name.startswith(("hipblasLt", "hipblasltExt")) or "hipblaslt_ext" in name:
                symbols.add(name)
    return symbols


def prepare(output, cache, flavor=None):
    sdk = find_sdk()
    flavor, version, profile = sdk_profile(sdk, flavor)
    sdk_library = sdk / "lib/libhipblaslt.so"
    dynamic = run("readelf", "-d", str(sdk_library), capture=True)
    soname = re.search(r"\(SONAME\).*\[(.*?)\]", dynamic)[1]
    if soname != profile["soname"]:
        raise RuntimeError(f"SDK SONAME {soname} differs from pinned {profile['soname']}")
    dependencies = re.findall(r"\(NEEDED\).*\[(.*?)\]", dynamic)
    targets, metadata = sdk_metadata(sdk, os.environ.get("GPU_TARGET"))
    metadata_hash = hashlib.sha256("".join(f"{name}\0{digest}\n" for name, digest in sorted(metadata.items())).encode()).hexdigest()
    patch_hash = hashlib.sha256("".join(f"{name}\0{sha256(ROOT / 'patches' / name)}\n" for name in profile["patches"]).encode()).hexdigest()
    inputs = {
        "source_repository": profile["source_repository"], "source_revision": profile["source_revision"],
        "msgpack_revision": MSGPACK_REVISION, "fmt_revision": FMT_REVISION, "yaml_revision": YAML_REVISION,
        "patch_sha256": patch_hash, "profiles_sha256": sha256(ROOT / "profiles.json"),
        "build_script_sha256": sha256(ROOT / "build.sh"), "prepare_script_sha256": sha256(__file__),
        "sdk_library_sha256": sha256(sdk_library), "kernel_metadata_sha256": metadata_hash,
        "gpu_targets": ",".join(targets), "sdk_dependencies": ",".join(sorted(dependencies)),
        "toolchain": run(str(sdk / "bin/amdclang++"), "--version", capture=True).strip().replace("\n", " | "),
        "cmake": run("cmake", "--version", capture=True).splitlines()[0],
        "flavor": flavor, "classifier": "linux-x86_64", "rocm_version": version,
        "hipblaslt_version": profile["hipblaslt_version"], "soname": soname, "library": soname,
    }
    for name in ("CFLAGS", "CXXFLAGS", "LDFLAGS", "CPLUS_INCLUDE_PATH"):
        inputs[f"build_env.{name}"] = os.environ.get(name, "")
    fingerprint = hashlib.sha256(repr(sorted(inputs.items())).encode()).hexdigest()
    artifact = cache / "artifacts" / fingerprint
    manifest = artifact / "hipblaslt.properties"
    expected = inputs | {f"kernel_metadata.{name}": digest for name, digest in metadata.items()}
    cached = properties(manifest) if manifest.is_file() else {}
    complete = all(cached.get(key) == value for key, value in expected.items())
    listed_files = {key[len("file_sha256."):]: digest for key, digest in cached.items() if key.startswith("file_sha256.")}
    complete = complete and soname in listed_files and any(name.startswith("licenses/") for name in listed_files)
    complete = complete and all((artifact / name).is_file() and sha256(artifact / name) == digest for name, digest in listed_files.items())
    if complete:
        print(f"Reusing patched hipBLASLt {flavor} {fingerprint}", flush=True)
    else:
        source = cache / "source" / f"{profile['source_revision']}-{patch_hash}"
        checkout_source(source, profile)
        apply_patches(source, profile)
        build = cache / "work" / fingerprint
        build_env = dict(os.environ, ROCM_PATH=str(sdk), HIPBLASLT_DEPS_DIR=str(build / "deps"),
                         GPU_TARGET=";".join(targets))
        run("bash", str(ROOT / "build.sh"), str(source), str(build), "--flavor", flavor, env=build_env)
        library = build / "library/libhipblaslt.so"
        actual_soname = re.search(r"\(SONAME\).*\[(.*?)\]", run("readelf", "-d", str(library), capture=True))[1]
        if actual_soname != soname:
            raise RuntimeError(f"Patched library SONAME {actual_soname} differs from SDK {soname}")
        missing_symbols = public_symbols(sdk_library) - public_symbols(library)
        if missing_symbols:
            raise RuntimeError("Patched hipBLASLt is missing SDK public symbols: " + ", ".join(sorted(missing_symbols)))
        artifact.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library, artifact / soname)
        licenses = artifact / "licenses"
        licenses.mkdir(exist_ok=True)
        originals = {"hipblaslt-LICENSE.md": source / profile["source_dir"] / "LICENSE.md",
                     "msgpack-COPYING": build / "deps/msgpack-cxx-6.1.0/source/COPYING",
                     "msgpack-LICENSE_1_0.txt": build / "deps/msgpack-cxx-6.1.0/source/LICENSE_1_0.txt"}
        if profile["build_family"] == "modern":
            originals["origami-LICENSE.md"] = source / "shared/origami/LICENSE.md"
        if any("rocroller" in name for name in dependencies):
            originals["fmt-LICENSE"] = build / "deps/fmt/source/LICENSE"
            originals["yaml-cpp-LICENSE"] = build / "deps/yaml-cpp/source/LICENSE"
        for name, original in originals.items():
            shutil.copy2(original, licenses / name)
        file_hashes = {f"file_sha256.{name}": sha256(artifact / name) for name in (soname, *(f"licenses/{name}" for name in originals))}
        write_properties(manifest, expected | {"sha256": sha256(artifact / soname)} | file_hashes)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "licenses").exists():
        shutil.rmtree(output / "licenses")
    for old in output.glob("libhipblaslt.so*"):
        old.unlink()
    shutil.copytree(artifact / "licenses", output / "licenses")
    shutil.copy2(artifact / soname, output / soname)
    shutil.copy2(manifest, output / manifest.name)
    print(f"Prepared private {flavor} hipBLASLt runtime: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--flavor", choices=PROFILES)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("The pinned runtime supports Linux x86_64 only")
    import fcntl
    cache = Path(os.environ.get("HIPBLASLT_CACHE_DIR", str(ROOT.parents[1] / "build/hipblaslt-cache"))).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare(args.output, cache, args.flavor)


if __name__ == "__main__":
    main()
