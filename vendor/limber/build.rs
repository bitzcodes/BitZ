#[path = "../../scripts/git_revision.rs"]
mod git_revision;

fn main() {
    let vendor = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    git_revision::metadata(&vendor.join("../.."));
}
