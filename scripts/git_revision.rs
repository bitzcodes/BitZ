use std::{collections::BTreeMap, path::Path, process::Command};

/// The same verifier is used by the build, materializer, and release packager.
pub fn metadata(root: &Path) -> BTreeMap<String, String> {
    let output = Command::new("python3")
        .arg(root.join("scripts/build_metadata.py"))
        .arg(root)
        .output()
        .expect("Python 3.11 or newer is required to verify vendor source");
    assert!(
        output.status.success(),
        "source provenance verification failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).expect("UTF-8 build metadata");
    let mut values = BTreeMap::new();
    for line in text.lines() {
        println!("{line}");
        if let Some(value) = line.strip_prefix("cargo:rustc-env=") {
            let (key, value) = value.split_once('=').expect("key=value build metadata");
            values.insert(key.to_owned(), value.to_owned());
        }
    }
    values
}
